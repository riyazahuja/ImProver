/-- The category of sheaves of modules over a scheme. -/
abbrev Modules := SheafOfModules.{u} X.ringCatSheaf


noncomputable instance : Abelian X.Modules := inferInstance


