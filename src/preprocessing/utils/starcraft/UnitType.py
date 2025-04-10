from enum import Enum, auto


class Category(Enum):
    UNIT = auto()
    TERRAN = auto()
    PROTOSS = auto()
    ZERG = auto()
    GROUND = auto()
    AIR = auto()
    BUILDING = auto()
    ADDON = auto()
    WORKER = auto()
    TRIVIAL = auto()
    SPELL = auto()
    RESOURCE = auto()


class UnitType(Enum):
    # Terran Units
    Terran_Marine = {Category.UNIT, Category.TERRAN, Category.GROUND}
    Terran_Firebat = {Category.UNIT, Category.TERRAN, Category.GROUND}
    Terran_Ghost = {Category.UNIT, Category.TERRAN, Category.GROUND}
    Terran_Goliath = {Category.UNIT, Category.TERRAN, Category.GROUND}
    Terran_Medic = {Category.UNIT, Category.TERRAN, Category.GROUND}
    Terran_SCV = {Category.UNIT, Category.TERRAN, Category.GROUND, Category.WORKER}
    Terran_Siege_Tank_Tank_Mode = {Category.UNIT, Category.TERRAN, Category.GROUND}
    Terran_Siege_Tank_Siege_Mode = {Category.UNIT, Category.TERRAN, Category.GROUND}
    Terran_Vulture = {Category.UNIT, Category.TERRAN, Category.GROUND}
    Terran_Vulture_Spider_Mine = {Category.UNIT, Category.TERRAN, Category.GROUND}
    Terran_Battlecruiser = {Category.UNIT, Category.TERRAN, Category.AIR}
    Terran_Dropship = {Category.UNIT, Category.TERRAN, Category.AIR}
    Terran_Nuclear_Missile = {Category.UNIT, Category.TERRAN}
    Terran_Science_Vessel = {Category.UNIT, Category.TERRAN, Category.AIR}
    Terran_Valkyrie = {Category.UNIT, Category.TERRAN, Category.AIR}
    Terran_Wraith = {Category.UNIT, Category.TERRAN, Category.AIR}

    # Terran Buildings
    Terran_Command_Center = {Category.BUILDING, Category.TERRAN}
    Terran_Barracks = {Category.BUILDING, Category.TERRAN}
    Terran_Bunker = {Category.BUILDING, Category.TERRAN}
    Terran_Engineering_Bay = {Category.BUILDING, Category.TERRAN}
    Terran_Missile_Turret = {Category.BUILDING, Category.TERRAN}
    Terran_Academy = {Category.BUILDING, Category.TERRAN}
    Terran_Armory = {Category.BUILDING, Category.TERRAN}
    Terran_Factory = {Category.BUILDING, Category.TERRAN}
    Terran_Refinery = {Category.BUILDING, Category.TERRAN}
    Terran_Science_Facility = {Category.BUILDING, Category.TERRAN}
    Terran_Starport = {Category.BUILDING, Category.TERRAN}
    Terran_Supply_Depot = {Category.BUILDING, Category.TERRAN}

    # Terran Add-ons
    Terran_Comsat_Station = {Category.ADDON, Category.BUILDING, Category.TERRAN}
    Terran_Control_Tower = {Category.ADDON, Category.BUILDING, Category.TERRAN}
    Terran_Covert_Ops = {Category.ADDON, Category.BUILDING, Category.TERRAN}
    Terran_Machine_Shop = {Category.ADDON, Category.BUILDING, Category.TERRAN}
    Terran_Nuclear_Silo = {Category.ADDON, Category.BUILDING, Category.TERRAN}
    Terran_Physics_Lab = {Category.ADDON, Category.BUILDING, Category.TERRAN}

    # Protoss Units
    Protoss_Zealot = {Category.UNIT, Category.PROTOSS, Category.GROUND}
    Protoss_Dragoon = {Category.UNIT, Category.PROTOSS, Category.GROUND}
    Protoss_Dark_Templar = {Category.UNIT, Category.PROTOSS, Category.GROUND}
    Protoss_High_Templar = {Category.UNIT, Category.PROTOSS, Category.GROUND}
    Protoss_Archon = {Category.UNIT, Category.PROTOSS, Category.GROUND}
    Protoss_Dark_Archon = {Category.UNIT, Category.PROTOSS, Category.GROUND}
    Protoss_Reaver = {Category.UNIT, Category.PROTOSS, Category.GROUND}
    Protoss_Scarab = {Category.UNIT, Category.PROTOSS}
    Protoss_Carrier = {Category.UNIT, Category.PROTOSS, Category.AIR}
    Protoss_Interceptor = {Category.UNIT, Category.PROTOSS, Category.AIR}
    Protoss_Arbiter = {Category.UNIT, Category.PROTOSS, Category.AIR}
    Protoss_Corsair = {Category.UNIT, Category.PROTOSS, Category.AIR}
    Protoss_Observer = {Category.UNIT, Category.PROTOSS, Category.AIR}
    Protoss_Scout = {Category.UNIT, Category.PROTOSS, Category.AIR}
    Protoss_Shuttle = {Category.UNIT, Category.PROTOSS, Category.AIR}
    Protoss_Probe = {Category.UNIT, Category.PROTOSS, Category.GROUND, Category.WORKER}

    # Protoss Buildings
    Protoss_Nexus = {Category.BUILDING, Category.PROTOSS}
    Protoss_Pylon = {Category.BUILDING, Category.PROTOSS}
    Protoss_Assimilator = {Category.BUILDING, Category.PROTOSS}
    Protoss_Gateway = {Category.BUILDING, Category.PROTOSS}
    Protoss_Forge = {Category.BUILDING, Category.PROTOSS}
    Protoss_Photon_Cannon = {Category.BUILDING, Category.PROTOSS}
    Protoss_Cybernetics_Core = {Category.BUILDING, Category.PROTOSS}
    Protoss_Citadel_of_Adun = {Category.BUILDING, Category.PROTOSS}
    Protoss_Templar_Archives = {Category.BUILDING, Category.PROTOSS}
    Protoss_Robotics_Facility = {Category.BUILDING, Category.PROTOSS}
    Protoss_Robotics_Support_Bay = {Category.BUILDING, Category.PROTOSS}
    Protoss_Observatory = {Category.BUILDING, Category.PROTOSS}
    Protoss_Stargate = {Category.BUILDING, Category.PROTOSS}
    Protoss_Fleet_Beacon = {Category.BUILDING, Category.PROTOSS}
    Protoss_Arbiter_Tribunal = {Category.BUILDING, Category.PROTOSS}

    # Zerg Units
    Zerg_Larva = {Category.UNIT, Category.ZERG, Category.TRIVIAL, Category.GROUND}
    Zerg_Zergling = {Category.UNIT, Category.ZERG, Category.GROUND}
    Zerg_Hydralisk = {Category.UNIT, Category.ZERG, Category.GROUND}
    Zerg_Lurker = {Category.UNIT, Category.ZERG, Category.GROUND}
    Zerg_Ultralisk = {Category.UNIT, Category.ZERG, Category.GROUND}
    Zerg_Broodling = {Category.UNIT, Category.ZERG, Category.GROUND}
    Zerg_Defiler = {Category.UNIT, Category.ZERG, Category.GROUND}
    Zerg_Queen = {Category.UNIT, Category.ZERG, Category.AIR}
    Zerg_Mutalisk = {Category.UNIT, Category.ZERG, Category.AIR}
    Zerg_Guardian = {Category.UNIT, Category.ZERG, Category.AIR}
    Zerg_Devourer = {Category.UNIT, Category.ZERG, Category.AIR}
    Zerg_Overlord = {Category.UNIT, Category.ZERG, Category.AIR}
    Zerg_Scourge = {Category.UNIT, Category.ZERG, Category.AIR}
    Zerg_Drone = {Category.UNIT, Category.ZERG, Category.GROUND, Category.WORKER}
    
    # Zerg Buildings
    Zerg_Hatchery = {Category.BUILDING, Category.ZERG}
    Zerg_Lair = {Category.BUILDING, Category.ZERG}
    Zerg_Hive = {Category.BUILDING, Category.ZERG}
    Zerg_Creep_Colony = {Category.BUILDING, Category.ZERG}
    Zerg_Spawning_Pool = {Category.BUILDING, Category.ZERG}
    Zerg_Evolution_Chamber = {Category.BUILDING, Category.ZERG}
    Zerg_Hydralisk_Den = {Category.BUILDING, Category.ZERG}
    Zerg_Spire = {Category.BUILDING, Category.ZERG}
    Zerg_Greater_Spire = {Category.BUILDING, Category.ZERG}
    Zerg_Queens_Nest = {Category.BUILDING, Category.ZERG}
    Zerg_Defiler_Mound = {Category.BUILDING, Category.ZERG}
    Zerg_Ultralisk_Cavern = {Category.BUILDING, Category.ZERG}
    Zerg_Nydus_Canal = {Category.BUILDING, Category.ZERG}
    Zerg_Extractor = {Category.BUILDING, Category.ZERG}
    Zerg_Sunken_Colony = {Category.BUILDING, Category.ZERG}
    Zerg_Spore_Colony = {Category.BUILDING, Category.ZERG}

    # Resource Units
    Resource_Mineral_Field = {Category.RESOURCE}
    Resource_Mineral_Field_Type_2 = {Category.RESOURCE}
    Resource_Mineral_Field_Type_3 = {Category.RESOURCE}
    Resource_Vespene_Geyser = {Category.RESOURCE}

    # Trivial Units (Critters)
    Critter_Bengalaas = {Category.UNIT, Category.TRIVIAL, Category.GROUND}
    Critter_Kakaru = {Category.UNIT, Category.TRIVIAL, Category.AIR}
    Critter_Ragnasaur = {Category.UNIT, Category.TRIVIAL, Category.GROUND}
    Critter_Rhynadon = {Category.UNIT, Category.TRIVIAL, Category.GROUND}
    Critter_Scantid = {Category.UNIT, Category.TRIVIAL, Category.GROUND}
    Critter_Ursadon = {Category.UNIT, Category.TRIVIAL, Category.GROUND}
    
    # Spells
    Spell_Dark_Swarm = {Category.SPELL}
    Spell_Disruption_Web = {Category.SPELL}
    Spell_Scanner_Sweep = {Category.SPELL}


    def belongs_to(self, *categories: Category) -> bool:
        """ 특정 유닛이 주어진 하나 이상의 카테고리에 속하는지 확인 """
        return any(category in self.value for category in categories)

    @classmethod
    def is_in_category(cls, name: str, category: Category) -> bool:
        """ 주어진 이름이 특정 카테고리에 속하는지 확인 """
        try:
            return category in cls[name].value
        except KeyError:
            return False
