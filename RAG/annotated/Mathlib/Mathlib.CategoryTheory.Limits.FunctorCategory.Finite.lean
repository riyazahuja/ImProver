instance [HasFiniteLimits C] : HasFiniteLimits (K ⥤ C) := ⟨fun _ ↦ inferInstance⟩


instance [HasFiniteProducts C] : HasFiniteProducts (K ⥤ C) := ⟨inferInstance⟩


instance [HasFiniteColimits C] : HasFiniteColimits (K ⥤ C) := ⟨fun _ ↦ inferInstance⟩


instance [HasFiniteCoproducts C] : HasFiniteCoproducts (K ⥤ C) := ⟨inferInstance⟩


