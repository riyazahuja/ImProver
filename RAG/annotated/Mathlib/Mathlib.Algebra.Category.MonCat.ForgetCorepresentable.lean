/-- The equivalence `(β →* γ) ≃ (α →* γ)` obtained by precomposition with
a multiplicative equivalence `e : α ≃* β`. -/
@[simps]
def precompEquiv {α β : Type*} [Monoid α] [Monoid β] (e : α ≃* β) (γ : Type*) [Monoid γ] :
    (β →* γ) ≃ (α →* γ) where
  toFun f := f.comp e
  invFun g := g.comp e.symm
                   /-
                     α : Type u_1
                     β : Type u_2
                     inst✝² : Monoid α
                     inst✝¹ : Monoid β
                     e : MulEquiv α β
                     γ : Type u_3
                     inst✝ : Monoid γ
                     x✝ : MonoidHom β γ
                     ⊢ Eq ((fun g => g.comp ↑e.symm) ((fun f => f.comp ↑e) x✝)) x✝
                   -/
  left_inv _ := by ext; simp
                        /-
                          🎉 no goals
                        -/
                    /-
                      α : Type u_1
                      β : Type u_2
                      inst✝² : Monoid α
                      inst✝¹ : Monoid β
                      e : MulEquiv α β
                      γ : Type u_3
                      inst✝ : Monoid γ
                      x✝ : MonoidHom α γ
                      ⊢ Eq ((fun f => f.comp ↑e) ((fun g => g.comp ↑e.symm) x✝)) x✝
                    -/
  right_inv _ := by ext; simp
                         /-
                           🎉 no goals
                         -/


/-- The equivalence `(Multiplicative ℕ →* α) ≃ α` for any monoid `α`. -/
@[simps]
def fromMultiplicativeNatEquiv (α : Type u) [Monoid α] : (Multiplicative ℕ →* α) ≃ α where
  toFun φ := φ (Multiplicative.ofAdd 1)
  invFun x := powersHom α x
                   /-
                     α : Type u
                     inst✝ : Monoid α
                     φ : MonoidHom (Multiplicative Nat) α
                     ⊢ Eq ((fun x => (powersHom α) x) ((fun φ => φ (Multiplicative.ofAdd 1)) φ)) φ
                   -/
  left_inv φ := by ext; simp
                        /-
                          🎉 no goals
                        -/
                    /-
                      α : Type u
                      inst✝ : Monoid α
                      x : α
                      ⊢ Eq ((fun φ => φ (Multiplicative.ofAdd 1)) ((fun x => (powersHom α) x) x)) x
                    -/
  right_inv x := by simp
                    /-
                      🎉 no goals
                    -/


/-- The equivalence `(ULift (Multiplicative ℕ) →* α) ≃ α` for any monoid `α`. -/
@[simps!]
def fromULiftMultiplicativeNatEquiv (α : Type u) [Monoid α] :
    (ULift.{u} (Multiplicative ℕ) →* α) ≃ α :=
  (precompEquiv (MulEquiv.ulift.symm) _).trans (fromMultiplicativeNatEquiv α)


/-- The equivalence `(β →+ γ) ≃ (α →+ γ)` obtained by precomposition with
an additive equivalence `e : α ≃+ β`. -/
@[simps]
def precompEquiv {α β : Type*} [AddMonoid α] [AddMonoid β] (e : α ≃+ β) (γ : Type*) [AddMonoid γ] :
    (β →+ γ) ≃ (α →+ γ) where
  toFun f := f.comp e
  invFun g := g.comp e.symm
                   /-
                     α : Type u_1
                     β : Type u_2
                     inst✝² : AddMonoid α
                     inst✝¹ : AddMonoid β
                     e : AddEquiv α β
                     γ : Type u_3
                     inst✝ : AddMonoid γ
                     x✝ : AddMonoidHom β γ
                     ⊢ Eq ((fun g => g.comp ↑e.symm) ((fun f => f.comp ↑e) x✝)) x✝
                   -/
  left_inv _ := by ext; simp
                        /-
                          🎉 no goals
                        -/
                    /-
                      α : Type u_1
                      β : Type u_2
                      inst✝² : AddMonoid α
                      inst✝¹ : AddMonoid β
                      e : AddEquiv α β
                      γ : Type u_3
                      inst✝ : AddMonoid γ
                      x✝ : AddMonoidHom α γ
                      ⊢ Eq ((fun f => f.comp ↑e) ((fun g => g.comp ↑e.symm) x✝)) x✝
                    -/
  right_inv _ := by ext; simp
                         /-
                           🎉 no goals
                         -/


/-- The equivalence `(ℤ →+ α) ≃ α` for any additive group `α`. -/
@[simps]
def fromNatEquiv (α : Type u) [AddMonoid α] : (ℕ →+ α) ≃ α where
  toFun φ := φ 1
  invFun x := multiplesHom α x
                   /-
                     α : Type u
                     inst✝ : AddMonoid α
                     φ : AddMonoidHom Nat α
                     ⊢ Eq ((fun x => (multiplesHom α) x) ((fun φ => φ 1) φ)) φ
                   -/
  left_inv φ := by ext; simp
                        /-
                          🎉 no goals
                        -/
                    /-
                      α : Type u
                      inst✝ : AddMonoid α
                      x : α
                      ⊢ Eq ((fun φ => φ 1) ((fun x => (multiplesHom α) x) x)) x
                    -/
  right_inv x := by simp
                    /-
                      🎉 no goals
                    -/


/-- The equivalence `(ULift ℕ →+ α) ≃ α` for any additive monoid `α`. -/
@[simps!]
def fromULiftNatEquiv (α : Type u) [AddMonoid α] : (ULift.{u} ℕ →+ α) ≃ α :=
  (precompEquiv (AddEquiv.ulift.symm) _).trans (fromNatEquiv α)


/-- The forgetful functor `MonCat.{u} ⥤ Type u` is corepresentable. -/
def MonCat.coyonedaObjIsoForget :
    coyoneda.obj (op (of (ULift.{u} (Multiplicative ℕ)))) ≅ forget MonCat.{u} :=
   /-
     ⊢ ∀ {X Y : MonCat} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.com …
   -/
  (NatIso.ofComponents (fun M => (MonoidHom.fromULiftMultiplicativeNatEquiv M.α).toIso))
   /-
     🎉 no goals
   -/



/-- The forgetful functor `CommMonCat.{u} ⥤ Type u` is corepresentable. -/
def CommMonCat.coyonedaObjIsoForget :
    coyoneda.obj (op (of (ULift.{u} (Multiplicative ℕ)))) ≅ forget CommMonCat.{u} :=
   /-
     ⊢ ∀ {X Y : CommMonCat} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct …
   -/
  (NatIso.ofComponents (fun M => (MonoidHom.fromULiftMultiplicativeNatEquiv M.α).toIso))
   /-
     🎉 no goals
   -/


/-- The forgetful functor `AddMonCat.{u} ⥤ Type u` is corepresentable. -/
def AddMonCat.coyonedaObjIsoForget :
    coyoneda.obj (op (of (ULift.{u} ℕ))) ≅ forget AddMonCat.{u} :=
   /-
     ⊢ ∀ {X Y : AddMonCat} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct. …
   -/
  (NatIso.ofComponents (fun M => (AddMonoidHom.fromULiftNatEquiv M.α).toIso))
   /-
     🎉 no goals
   -/


/-- The forgetful functor `AddCommMonCat.{u} ⥤ Type u` is corepresentable. -/
def AddCommMonCat.coyonedaObjIsoForget :
    coyoneda.obj (op (of (ULift.{u} ℕ))) ≅ forget AddCommMonCat.{u} :=
   /-
     ⊢ ∀ {X Y : AddCommMonCat} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStr …
   -/
  (NatIso.ofComponents (fun M => (AddMonoidHom.fromULiftNatEquiv M.α).toIso))
   /-
     🎉 no goals
   -/


instance MonCat.forget_isCorepresentable :
    (forget MonCat.{u}).IsCorepresentable :=
  Functor.IsCorepresentable.mk' MonCat.coyonedaObjIsoForget


instance CommMonCat.forget_isCorepresentable :
    (forget CommMonCat.{u}).IsCorepresentable :=
  Functor.IsCorepresentable.mk' CommMonCat.coyonedaObjIsoForget


instance AddMonCat.forget_isCorepresentable :
    (forget AddMonCat.{u}).IsCorepresentable :=
  Functor.IsCorepresentable.mk' AddMonCat.coyonedaObjIsoForget


instance AddCommMonCat.forget_isCorepresentable :
    (forget AddCommMonCat.{u}).IsCorepresentable :=
  Functor.IsCorepresentable.mk' AddCommMonCat.coyonedaObjIsoForget

