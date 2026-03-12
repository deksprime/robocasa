import math
import os
import xml.etree.ElementTree as ET
from copy import deepcopy

import numpy as np
from robosuite.utils.mjcf_utils import find_elements, string_to_array

import robocasa
from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES, OBJ_GROUPS

BASE_ASSET_ZOO_PATH = os.path.join(robocasa.models.assets_root, "objects")
_ASSET_ZOO_PATHS = robocasa.models.get_asset_paths("objects")


class ObjCat:
    """
    Class that encapsulates data for an object category.

    Args:
        name (str): name of the object category

        types (tuple) or (str): type(s)/categories the object belongs to. Examples include meat, sweets, fruit, etc.

        model_folders (list): list of folders containing the MJCF models for the object category

        exclude (list): list of model names to exclude

        graspable (bool): whether the object is graspable

        washable (bool): whether the object is washable

        microwavable (bool): whether the object is microwavable

        cookable (bool): whether the object is cookable

        fridgable (bool): whether the object is fridgable

        freezable (bool): whether the object is freezable

        dishwashable (bool): whether the object is dishwashable

        scale (float): scale of the object meshes/geoms

        solimp (tuple): solimp values for the object meshes/geoms

        solref (tuple): solref values for the object meshes/geoms

        density (float): density of the object meshes/geoms

        friction (tuple): friction values for the object meshes/geoms

        priority: priority of the object

        reg_type (str): registry type (options: objaverse, aigen_objs, lightwheel)

        auxiliary_obj (str): name of the auxiliary object group (e.g., "saucepan_lid")

        is_auxiliary_obj (bool): if True, models are located one directory deeper
    """

    def __init__(
        self,
        name,
        types,
        model_folders=None,
        exclude=None,
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        fridgable=False,
        freezable=False,
        dishwashable=False,
        scale=1.0,
        solimp=(0.998, 0.998, 0.001),
        solref=(0.001, 2),
        density=100,
        friction=(0.95, 0.3, 0.1),
        priority=None,
        reg_type="objaverse",
        auxiliary_obj=None,
        is_auxiliary_obj=False,
    ):
        self.name = name
        if not isinstance(types, tuple):
            types = (types,)
        self.types = types

        self.reg_type = reg_type

        self.graspable = graspable
        self.washable = washable
        self.microwavable = microwavable
        self.cookable = cookable
        self.fridgable = fridgable
        self.freezable = freezable
        self.dishwashable = dishwashable
        self.auxiliary_obj = auxiliary_obj
        self.is_auxiliary_obj = is_auxiliary_obj

        self.scale = scale
        self.solimp = solimp
        self.solref = solref
        self.density = density
        self.friction = friction
        self.priority = priority
        self.exclude = exclude or []

        if model_folders is None:
            model_folders = ["{}/{}".format(reg_type, name)]
        cat_mjcf_paths = []
        # Search both built-in and extension asset directories
        for zoo_path in _ASSET_ZOO_PATHS:
            for folder in model_folders:
                cat_path = os.path.join(zoo_path, folder)
                if not os.path.exists(cat_path):
                    continue
                for model_name in os.listdir(cat_path):
                    model_dir = os.path.join(cat_path, model_name)
                    if not os.path.isdir(model_dir):
                        continue
                    if self.is_auxiliary_obj:
                        for sub_name in os.listdir(model_dir):
                            sub_dir = os.path.join(model_dir, sub_name)
                            if os.path.isdir(sub_dir) and "model.xml" in os.listdir(
                                sub_dir
                            ):
                                if model_name in self.exclude:
                                    continue
                                cat_mjcf_paths.append(os.path.join(sub_dir, "model.xml"))
                    else:
                        if "model.xml" in os.listdir(model_dir):
                            if model_name in self.exclude:
                                continue
                            cat_mjcf_paths.append(os.path.join(model_dir, "model.xml"))
        self.mjcf_paths = sorted(cat_mjcf_paths)

    def get_mjcf_kwargs(self):
        """
        returns relevant data to apply to the MJCF model for the object category
        """
        return deepcopy(
            dict(
                scale=self.scale,
                solimp=self.solimp,
                solref=self.solref,
                density=self.density,
                friction=self.friction,
                priority=self.priority,
            )
        )


# update OBJ_CATEGORIES with ObjCat instances. Maps name to the different registries it can belong to
# and then maps the registry to the ObjCat instance
for (name, kwargs) in OBJ_CATEGORIES.items():

    # get the properties that are common to both registries
    common_properties = deepcopy(kwargs)
    for k in common_properties.keys():
        assert k in [
            "graspable",
            "washable",
            "microwavable",
            "cookable",
            "fridgable",
            "freezable",
            "auxiliary_obj",
            "is_auxiliary_obj",
            "dishwashable",
            "types",
            "aigen",
            "objaverse",
            "lightwheel",
        ]
    objaverse_kwargs = common_properties.pop("objaverse", None)
    aigen_kwargs = common_properties.pop("aigen", None)
    lightwheel_kwargs = common_properties.pop("lightwheel", None)
    assert "scale" not in kwargs
    OBJ_CATEGORIES[name] = {}

    # create instances
    if objaverse_kwargs is not None:
        objaverse_kwargs.update(common_properties)
        OBJ_CATEGORIES[name]["objaverse"] = ObjCat(
            name=name, reg_type="objaverse", **objaverse_kwargs
        )
    if aigen_kwargs is not None:
        aigen_kwargs.update(common_properties)
        OBJ_CATEGORIES[name]["aigen"] = ObjCat(
            name=name, reg_type="aigen_objs", **aigen_kwargs
        )
    if lightwheel_kwargs is not None:
        lightwheel_kwargs.update(common_properties)
        OBJ_CATEGORIES[name]["lightwheel"] = ObjCat(
            name=name, reg_type="lightwheel", **lightwheel_kwargs
        )

# Load extension object categories.
# Extensions can define objects by placing YAML category definitions at:
#   {extensions_root}/assets/objects/categories.yaml
# Or simply by placing model.xml dirs under:
#   {extensions_root}/assets/objects/{registry}/{category_name}/{model_name}/model.xml
# Auto-discovered categories use "custom" registry type with default properties.
_ext_objects_dir = os.path.join(robocasa.models.extensions_assets, "objects")
if os.path.isdir(_ext_objects_dir):
    import json as _json
    import yaml as _yaml

    # Load explicit category definitions if present
    for _cfg_name in ("categories.yaml", "categories.json"):
        _cfg_path = os.path.join(_ext_objects_dir, _cfg_name)
        if os.path.isfile(_cfg_path):
            with open(_cfg_path) as _f:
                _ext_cats = _yaml.safe_load(_f) if _cfg_name.endswith(".yaml") else _json.load(_f)
            for _name, _kwargs in (_ext_cats or {}).items():
                if _name not in OBJ_CATEGORIES:
                    OBJ_CATEGORIES[_name] = {}
                    if _name not in OBJ_GROUPS.get("all", []):
                        OBJ_GROUPS.setdefault("all", []).append(_name)
                _reg = _kwargs.pop("reg_type", "custom")
                OBJ_CATEGORIES[_name][_reg] = ObjCat(name=_name, reg_type=_reg, **_kwargs)

    # Auto-discover: scan for registry dirs that contain category subdirs with model.xml
    for _reg_name in os.listdir(_ext_objects_dir):
        _reg_dir = os.path.join(_ext_objects_dir, _reg_name)
        if not os.path.isdir(_reg_dir) or _reg_name.startswith(".") or _reg_name.endswith((".yaml", ".json")):
            continue
        for _cat_name in os.listdir(_reg_dir):
            _cat_dir = os.path.join(_reg_dir, _cat_name)
            if not os.path.isdir(_cat_dir):
                continue
            # Check if this category has any model.xml files
            _has_models = any(
                os.path.isfile(os.path.join(_cat_dir, m, "model.xml"))
                for m in os.listdir(_cat_dir)
                if os.path.isdir(os.path.join(_cat_dir, m))
            )
            if _has_models and _cat_name not in OBJ_CATEGORIES:
                OBJ_CATEGORIES[_cat_name] = {}
                OBJ_GROUPS.setdefault("all", []).append(_cat_name)
            if _has_models and _reg_name not in OBJ_CATEGORIES.get(_cat_name, {}):
                OBJ_CATEGORIES[_cat_name][_reg_name] = ObjCat(
                    name=_cat_name,
                    reg_type=_reg_name,
                    types=("custom",),
                    graspable=True,
                    scale=1.0,
                )


def sample_kitchen_object(
    groups,
    exclude_groups=None,
    graspable=None,
    washable=None,
    microwavable=None,
    cookable=None,
    fridgable=None,
    freezable=None,
    dishwashable=None,
    auxiliary_obj=None,
    rng=None,
    obj_registries=("objaverse", "lightwheel"),
    split=None,
    max_size=(None, None, None),
    object_scale=None,
    rotate_upright=False,
):
    """
    Sample a kitchen object from the specified groups and within max_size bounds.

    Args:
        groups (list or str): groups to sample from or the exact xml path of the object to spawn

        exclude_groups (str or list): groups to exclude

        graspable (bool): whether the sampled object must be graspable

        washable (bool): whether the sampled object must be washable

        microwavable (bool): whether the sampled object must be microwavable

        cookable (bool): whether whether the sampled object must be cookable

        fridgable (bool): whether whether the sampled object must be fridgable

        freezable (bool): whether whether the sampled object must be freezable

        dishwashable (bool): whether whether the sampled object must be dishwashable

        auxiliary_obj (str): name of the auxiliary object group to look for

        rng (np.random.Generator): random number object

        obj_registries (tuple): registries to sample from

        split (str): split to sample from. Split "pretrain" specifies all but the last 4 object instances
                    (or the first half - whichever is larger), "target" specifies the rest, and None specifies all.

        max_size (tuple): max size of the object. If the sampled object is not within bounds of max size, function will resample

        object_scale (float): scale of the object. If set will multiply the scale of the sampled object by this value


    Returns:
        dict: kwargs to apply to the MJCF model for the sampled object

        dict: info about the sampled object - the path of the mjcf, groups which the object's category belongs to, the category of the object
              the sampling split the object came from, and the groups the object was sampled from
    """
    valid_object_sampled = False
    while valid_object_sampled is False:
        mjcf_kwargs, info = sample_kitchen_object_helper(
            groups=groups,
            exclude_groups=exclude_groups,
            graspable=graspable,
            washable=washable,
            microwavable=microwavable,
            cookable=cookable,
            fridgable=fridgable,
            freezable=freezable,
            dishwashable=dishwashable,
            auxiliary_obj=auxiliary_obj,
            rng=rng,
            obj_registries=obj_registries,
            split=split,
            object_scale=object_scale,
            rotate_upright=rotate_upright,
        )

        # check if object size is within bounds
        mjcf_path = info["mjcf_path"]
        tree = ET.parse(mjcf_path)
        root = tree.getroot()
        half_size = string_to_array(
            find_elements(root=root, tags="geom", attribs={"name": "reg_bbox"}).get(
                "size"
            )
        )
        scale = mjcf_kwargs["scale"]
        obj_size = (half_size * 2) * scale

        valid_object_sampled = True
        for i in range(3):
            if max_size[i] is not None and obj_size[i] > max_size[i]:
                valid_object_sampled = False

    return mjcf_kwargs, info


def sample_kitchen_object_helper(
    groups,
    exclude_groups=None,
    graspable=None,
    washable=None,
    microwavable=None,
    cookable=None,
    fridgable=None,
    freezable=None,
    dishwashable=None,
    auxiliary_obj=None,
    rng=None,
    obj_registries=("objaverse",),
    split=None,
    object_scale=None,
    rotate_upright=False,
):
    """
    Helper function to sample a kitchen object.

    Args:
        groups (list or str): groups to sample from or the exact xml path of the object to spawn

        exclude_groups (str or list): groups to exclude

        graspable (bool): whether the sampled object must be graspable

        washable (bool): whether the sampled object must be washable

        microwavable (bool): whether the sampled object must be microwavable

        cookable (bool): whether whether the sampled object must be cookable

        fridgable (bool): whether whether the sampled object must be fridgable

        freezable (bool): whether whether the sampled object must be freezable

        dishwashable (bool): whether whether the sampled object must be dishwashable

        auxiliary_obj (str): name of the auxiliary object group to look for

        rng (np.random.Generator): random number object

        obj_registries (tuple): registries to sample from

        split (str): split to sample from. Split "pretrain" specifies all but the last 4 object instances
                    (or the first half - whichever is larger), "target" specifies the rest, and None specifies all.

        object_scale (float): scale of the object. If set will multiply the scale of the sampled object by this value


    Returns:
        dict: kwargs to apply to the MJCF model for the sampled object

        dict: info about the sampled object - the path of the mjcf, groups which the object's category belongs to, the category of the object
              the sampling split the object came from, and the groups the object was sampled from
    """
    if rng is None:
        rng = np.random.default_rng()

    # option to spawn specific object instead of sampling from a group
    if isinstance(groups, str) and groups.endswith(".xml"):
        mjcf_path = groups
        model_xml_path = os.path.join(os.path.dirname(mjcf_path), "model.xml")
        # reverse look up mjcf_path to category
        mjcf_kwargs = dict()
        cat = None
        obj_found = False
        for cand_cat in OBJ_CATEGORIES:
            for reg in obj_registries:
                if (
                    reg in OBJ_CATEGORIES[cand_cat]
                    and model_xml_path in OBJ_CATEGORIES[cand_cat][reg].mjcf_paths
                ):
                    mjcf_kwargs = OBJ_CATEGORIES[cand_cat][reg].get_mjcf_kwargs()
                    cat = cand_cat
                    obj_found = True
                    break
            if obj_found:
                break
        if obj_found is False:
            raise ValueError
        mjcf_kwargs["mjcf_path"] = mjcf_path
    else:
        if not isinstance(groups, tuple) and not isinstance(groups, list):
            groups = [groups]

        if exclude_groups is None:
            exclude_groups = []
        if not isinstance(exclude_groups, tuple) and not isinstance(
            exclude_groups, list
        ):
            exclude_groups = [exclude_groups]

        invalid_categories = []
        for g in exclude_groups:
            for cat in OBJ_GROUPS[g]:
                invalid_categories.append(cat)

        valid_categories = []
        for g in groups:
            for cat in OBJ_GROUPS[g]:
                # don't repeat if already added
                if cat in valid_categories:
                    continue
                if cat in invalid_categories:
                    continue

                # don't include if category not represented in any registry
                cat_in_any_reg = np.any(
                    [reg in OBJ_CATEGORIES[cat] for reg in obj_registries]
                )
                if not cat_in_any_reg:
                    continue

                invalid = False
                for reg in obj_registries:
                    if reg not in OBJ_CATEGORIES[cat]:
                        continue
                    cat_meta = OBJ_CATEGORIES[cat][reg]
                    if graspable is True and cat_meta.graspable is not True:
                        invalid = True
                    if washable is True and cat_meta.washable is not True:
                        invalid = True
                    if microwavable is True and cat_meta.microwavable is not True:
                        invalid = True
                    if cookable is True and cat_meta.cookable is not True:
                        invalid = True
                    if fridgable is True and cat_meta.fridgable is not True:
                        invalid = True
                    if freezable is True and cat_meta.freezable is not True:
                        invalid = True
                    if dishwashable is True and cat_meta.dishwashable is not True:
                        invalid = True

                if invalid:
                    continue

                valid_categories.append(cat)

        cat = rng.choice(valid_categories)

        choices = {reg: [] for reg in obj_registries}

        for reg in obj_registries:
            if reg not in OBJ_CATEGORIES[cat]:
                choices[reg] = []
                continue
            reg_choices = deepcopy(OBJ_CATEGORIES[cat][reg].mjcf_paths)

            # exclude out objects based on split
            if split is not None:
                split_th = max(
                    len(reg_choices) - 5, int(math.ceil(len(reg_choices) / 2))
                )
                if split == "pretrain":
                    reg_choices = reg_choices[:split_th]
                elif split == "target":
                    reg_choices = reg_choices[split_th:]
                else:
                    raise ValueError
            choices[reg] = reg_choices

        chosen_reg = rng.choice(
            obj_registries,
            p=np.array([len(choices[reg]) for reg in obj_registries])
            / sum(len(choices[reg]) for reg in obj_registries),
        )

        mjcf_path = rng.choice(choices[chosen_reg])
        if rotate_upright:
            mjcf_path = mjcf_path.replace("model.xml", "model_upright.xml")
        mjcf_kwargs = OBJ_CATEGORIES[cat][chosen_reg].get_mjcf_kwargs()
        mjcf_kwargs["mjcf_path"] = mjcf_path

    if object_scale is not None:
        if isinstance(object_scale, float):
            if isinstance(mjcf_kwargs["scale"], float):
                mjcf_kwargs["scale"] *= object_scale
            else:
                mjcf_kwargs["scale"] = [e * object_scale for e in mjcf_kwargs["scale"]]
        else:
            if isinstance(mjcf_kwargs["scale"], float):
                mjcf_kwargs["scale"] = [mjcf_kwargs["scale"] for _ in range(3)]
            mjcf_kwargs["scale"] = [
                mjcf_kwargs["scale"][ind] * object_scale[ind] for ind in range(3)
            ]

    groups_containing_sampled_obj = []
    for group, group_cats in OBJ_GROUPS.items():
        if cat in group_cats:
            groups_containing_sampled_obj.append(group)

    info = {
        "groups_containing_sampled_obj": groups_containing_sampled_obj,
        "groups": groups,
        "cat": cat,
        "split": split,
        "mjcf_path": mjcf_path,
    }

    return mjcf_kwargs, info
