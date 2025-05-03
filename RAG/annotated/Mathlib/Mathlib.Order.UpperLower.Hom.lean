/-- `UpperSet.Ici` as a `SupHom`. -/
def iciSupHom : SupHom α (UpperSet α) :=
  ⟨Ici, Ici_sup⟩


@[simp]
theorem coe_iciSupHom : (iciSupHom : α → UpperSet α) = Ici :=
  rfl


@[simp]
theorem iciSupHom_apply (a : α) : iciSupHom a = Ici a :=
  rfl


/-- `UpperSet.Ici` as a `sSupHom`. -/
def icisSupHom : sSupHom α (UpperSet α) :=
  ⟨Ici, fun s => (Ici_sSup s).trans sSup_image.symm⟩
-- Porting note: `ₓ` because typeclass assumption changed


@[simp]
theorem coe_icisSupHom : (icisSupHom : α → UpperSet α) = Ici :=
  rfl
-- Porting note: `ₓ` because typeclass assumption changed


@[simp]
theorem icisSupHom_apply (a : α) : icisSupHom a = Ici a :=
  rfl
-- Porting note: `ₓ` because typeclass assumption changed


/-- `LowerSet.Iic` as an `InfHom`. -/
def iicInfHom : InfHom α (LowerSet α) :=
  ⟨Iic, Iic_inf⟩


@[simp]
theorem coe_iicInfHom : (iicInfHom : α → LowerSet α) = Iic :=
  rfl


@[simp]
theorem iicInfHom_apply (a : α) : iicInfHom a = Iic a :=
  rfl


/-- `LowerSet.Iic` as an `sInfHom`. -/
def iicsInfHom : sInfHom α (LowerSet α) :=
  ⟨Iic, fun s => (Iic_sInf s).trans sInf_image.symm⟩
-- Porting note: `ₓ` because typeclass assumption changed


@[simp]
theorem coe_iicsInfHom : (iicsInfHom : α → LowerSet α) = Iic :=
  rfl
-- Porting note: `ₓ` because typeclass assumption changed


@[simp]
theorem iicsInfHom_apply (a : α) : iicsInfHom a = Iic a :=
  rfl
-- Porting note: `ₓ` because typeclass assumption changed


