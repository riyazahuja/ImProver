/-- The equivalence `(Multiplicative ℤ →* α) ≃ α` for any group `α`. -/
@[simps]
def fromMultiplicativeIntEquiv (α : Type u) [Group α] : (Multiplicative ℤ →* α) ≃ α where
  toFun φ := φ (Multiplicative.ofAdd 1)
  invFun x := zpowersHom α x
                   /-
                     α : Type u
                     inst✝ : Group α
                     φ : MonoidHom (Multiplicative Int) α
                     ⊢ Eq ((fun x => (zpowersHom α) x) ((fun φ => φ (Multiplicative.ofAdd 1)) φ)) φ
                   -/
  left_inv φ := by ext; simp
                        /-
                          🎉 no goals
                        -/
                    /-
                      α : Type u
                      inst✝ : Group α
                      x : α
                      ⊢ Eq ((fun φ => φ (Multiplicative.ofAdd 1)) ((fun x => (zpowersHom α) x) x)) x
                    -/
  right_inv x := by simp
                    /-
                      🎉 no goals
                    -/


/-- The equivalence `(ULift (Multiplicative ℤ) →* α) ≃ α` for any group `α`. -/
@[simps!]
def fromULiftMultiplicativeIntEquiv (α : Type u) [Group α] :
    (ULift.{u} (Multiplicative ℤ) →* α) ≃ α :=
  (precompEquiv (MulEquiv.ulift.symm) _).trans (fromMultiplicativeIntEquiv α)


/-- The equivalence `(ℤ →+ α) ≃ α` for any additive group `α`. -/
@[simps]
def fromIntEquiv (α : Type u) [AddGroup α] : (ℤ →+ α) ≃ α where
  toFun φ := φ 1
  invFun x := zmultiplesHom α x
                   /-
                     α : Type u
                     inst✝ : AddGroup α
                     φ : AddMonoidHom Int α
                     ⊢ Eq ((fun x => (zmultiplesHom α) x) ((fun φ => φ 1) φ)) φ
                   -/
  left_inv φ := by ext; simp
                        /-
                          🎉 no goals
                        -/
                    /-
                      α : Type u
                      inst✝ : AddGroup α
                      x : α
                      ⊢ Eq ((fun φ => φ 1) ((fun x => (zmultiplesHom α) x) x)) x
                    -/
  right_inv x := by simp
                    /-
                      🎉 no goals
                    -/


/-- The equivalence `(ULift ℤ →+ α) ≃ α` for any additive group `α`. -/
@[simps!]
def fromULiftIntEquiv (α : Type u) [AddGroup α] : (ULift.{u} ℤ →+ α) ≃ α :=
  (precompEquiv (AddEquiv.ulift.symm) _).trans (fromIntEquiv α)


/-- The forget functor `Grp.{u} ⥤ Type u` is corepresentable. -/
def Grp.coyonedaObjIsoForget :
    coyoneda.obj (op (of (ULift.{u} (Multiplicative ℤ)))) ≅ forget Grp.{u} :=
   /-
     ⊢ ∀ {X Y : Grp} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ( …
   -/
  (NatIso.ofComponents (fun M => (MonoidHom.fromULiftMultiplicativeIntEquiv M.α).toIso))
   /-
     🎉 no goals
   -/


/-- The forget functor `CommGrp.{u} ⥤ Type u` is corepresentable. -/
def CommGrp.coyonedaObjIsoForget :
    coyoneda.obj (op (of (ULift.{u} (Multiplicative ℤ)))) ≅ forget CommGrp.{u} :=
   /-
     ⊢ ∀ {X Y : CommGrp} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.co …
   -/
  (NatIso.ofComponents (fun M => (MonoidHom.fromULiftMultiplicativeIntEquiv M.α).toIso))
   /-
     🎉 no goals
   -/


/-- The forget functor `AddGrp.{u} ⥤ Type u` is corepresentable. -/
def AddGrp.coyonedaObjIsoForget :
    coyoneda.obj (op (of (ULift.{u} ℤ))) ≅ forget AddGrp.{u} :=
   /-
     ⊢ ∀ {X Y : AddGrp} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.com …
   -/
  (NatIso.ofComponents (fun M => (AddMonoidHom.fromULiftIntEquiv M.α).toIso))
   /-
     🎉 no goals
   -/


/-- The forget functor `AddCommGrp.{u} ⥤ Type u` is corepresentable. -/
def AddCommGrp.coyonedaObjIsoForget :
    coyoneda.obj (op (of (ULift.{u} ℤ))) ≅ forget AddCommGrp.{u} :=
   /-
     ⊢ ∀ {X Y : AddCommGrp} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct …
   -/
  (NatIso.ofComponents (fun M => (AddMonoidHom.fromULiftIntEquiv M.α).toIso))
   /-
     🎉 no goals
   -/


instance Grp.forget_isCorepresentable :
    (forget Grp.{u}).IsCorepresentable :=
  Functor.IsCorepresentable.mk' Grp.coyonedaObjIsoForget


instance CommGrp.forget_isCorepresentable :
    (forget CommGrp.{u}).IsCorepresentable :=
  Functor.IsCorepresentable.mk' CommGrp.coyonedaObjIsoForget


instance AddGrp.forget_isCorepresentable :
    (forget AddGrp.{u}).IsCorepresentable :=
  Functor.IsCorepresentable.mk' AddGrp.coyonedaObjIsoForget


instance AddCommGrp.forget_isCorepresentable :
    (forget AddCommGrp.{u}).IsCorepresentable :=
  Functor.IsCorepresentable.mk' AddCommGrp.coyonedaObjIsoForget

