instance : HasLimitsOfSize.{u, u} LightCondSet.{u} := by
  /-
    ⊢ CategoryTheory.Limits.HasLimitsOfSize.{u, u, u + 1, u + 1} LightCondSet
  -/
  change HasLimitsOfSize (Sheaf _ _)
  /-
    ⊢ CategoryTheory.Limits.HasLimitsOfSize.{u, u, u + 1, u + 1} (CategoryTheory.S …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance : HasFiniteLimits LightCondSet.{u} := hasFiniteLimits_of_hasLimitsOfSize _


instance : HasLimitsOfSize.{u, u} (LightCondMod.{u} R) :=
  inferInstanceAs (HasLimitsOfSize (Sheaf _ _))


instance : HasLimitsOfSize.{0, 0} (LightCondMod.{u} R) :=
  inferInstanceAs (HasLimitsOfSize (Sheaf _ _))


instance : HasFiniteLimits (LightCondMod.{u} R) := hasFiniteLimits_of_hasLimitsOfSize _

