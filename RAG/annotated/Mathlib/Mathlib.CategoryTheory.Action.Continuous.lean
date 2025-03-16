instance : HasForget₂ (Action V G) TopCat :=
  HasForget₂.trans (Action V G) V TopCat


instance (X : Action V G) : MulAction G ((CategoryTheory.forget₂ _ TopCat).obj X) where
  smul g x := ((CategoryTheory.forget₂ _ TopCat).map (X.ρ g)) x
  one_smul x := by
    /-
      V : Type (u + 1)
      inst✝³ : CategoryTheory.LargeCategory V
      inst✝² : CategoryTheory.ConcreteCategory V
      inst✝¹ : CategoryTheory.HasForget₂ V TopCat
      G : MonCat
      inst✝ : TopologicalSpace ↑G
      X : Action V G
      x : ↑((CategoryTheory.forget₂ (Action V G) TopCat).obj X)
      ⊢ Eq (HSMul.hSMul 1 x) x
    -/
    show ((CategoryTheory.forget₂ _ TopCat).map (X.ρ 1)) x = x
    /-
      V : Type (u + 1)
      inst✝³ : CategoryTheory.LargeCategory V
      inst✝² : CategoryTheory.ConcreteCategory V
      inst✝¹ : CategoryTheory.HasForget₂ V TopCat
      G : MonCat
      inst✝ : TopologicalSpace ↑G
      X : Action V G
      x : ↑((CategoryTheory.forget₂ (Action V G) TopCat).obj X)
      ⊢ Eq (((CategoryTheory.forget₂ V TopCat).map (X.ρ 1)) x) x
    -/
    simp
    /-
      🎉 no goals
    -/
  mul_smul g h x := by
    show (CategoryTheory.forget₂ _ TopCat).map (X.ρ (g * h)) x =
      ((CategoryTheory.forget₂ _ TopCat).map (X.ρ h) ≫
        (CategoryTheory.forget₂ _ TopCat).map (X.ρ g)) x
    /-
      V : Type (u + 1)
      inst✝³ : CategoryTheory.LargeCategory V
      inst✝² : CategoryTheory.ConcreteCategory V
      inst✝¹ : CategoryTheory.HasForget₂ V TopCat
      G : MonCat
      inst✝ : TopologicalSpace ↑G
      X : Action V G
      g h : ↑G
      x : ↑((CategoryTheory.forget₂ (Action V G) TopCat).obj X)
      ⊢ Eq (((CategoryTheory.forget₂ V TopCat).map (X.ρ (HMul.hMul g h))) x) ((Categ …
    -/
    rw [← Functor.map_comp, map_mul]
    /-
      V : Type (u + 1)
      inst✝³ : CategoryTheory.LargeCategory V
      inst✝² : CategoryTheory.ConcreteCategory V
      inst✝¹ : CategoryTheory.HasForget₂ V TopCat
      G : MonCat
      inst✝ : TopologicalSpace ↑G
      X : Action V G
      g h : ↑G
      x : ↑((CategoryTheory.forget₂ (Action V G) TopCat).obj X)
      ⊢ Eq (((CategoryTheory.forget₂ V TopCat).map (HMul.hMul (X.ρ g) (X.ρ h))) x) ( …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- For `HasForget₂ V TopCat` a predicate on an `X : Action V G` saying that the induced action on
the underlying topological space is continuous. -/
abbrev IsContinuous (X : Action V G) : Prop :=
  ContinuousSMul G ((CategoryTheory.forget₂ _ TopCat).obj X)


/-- For `HasForget₂ V TopCat`, this is the full subcategory of `Action V G` where the induced
action is continuous. -/
def ContAction : Type _ := FullSubcategory (IsContinuous (V := V) (G := G))


instance : Category (ContAction V G) :=
  FullSubcategory.category (IsContinuous (V := V) (G := G))


instance : ConcreteCategory (ContAction V G) :=
  FullSubcategory.concreteCategory (IsContinuous (V := V) (G := G))


instance : HasForget₂ (ContAction V G) (Action V G) :=
  FullSubcategory.hasForget₂ (IsContinuous (V := V) (G := G))


instance : HasForget₂ (ContAction V G) V :=
  HasForget₂.trans (ContAction V G) (Action V G) V


instance : HasForget₂ (ContAction V G) TopCat :=
  HasForget₂.trans (ContAction V G) (Action V G) TopCat


instance : Coe (ContAction V G) (Action V G) where
  coe X := X.obj


/-- A predicate on an `X : ContAction V G` saying that the topology on the underlying type of `X`
is discrete. -/
abbrev IsDiscrete (X : ContAction V G) : Prop :=
  DiscreteTopology ((CategoryTheory.forget₂ _ TopCat).obj X)


/-- The subcategory of `ContAction V G` where the topology is discrete. -/
def DiscreteContAction : Type _ := FullSubcategory (IsDiscrete (V := V) (G := G))


instance : Category (DiscreteContAction V G) :=
  FullSubcategory.category (IsDiscrete (V := V) (G := G))


instance : ConcreteCategory (DiscreteContAction V G) :=
  FullSubcategory.concreteCategory (IsDiscrete (V := V) (G := G))


instance : HasForget₂ (DiscreteContAction V G) (ContAction V G) :=
  FullSubcategory.hasForget₂ (IsDiscrete (V := V) (G := G))


instance : HasForget₂ (DiscreteContAction V G) TopCat :=
  HasForget₂.trans (DiscreteContAction V G) (ContAction V G) TopCat


instance (X : DiscreteContAction V G) :
    DiscreteTopology ((CategoryTheory.forget₂ _ TopCat).obj X) :=
  X.property


