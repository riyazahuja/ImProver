/-- A typeclass for nonempty finite linear orders. -/
class NonemptyFiniteLinearOrder (α : Type*) extends Fintype α, LinearOrder α where
  Nonempty : Nonempty α := by infer_instance


instance (priority := 100) NonemptyFiniteLinearOrder.toBoundedOrder (α : Type*)
  [NonemptyFiniteLinearOrder α] : BoundedOrder α :=
  Fintype.toBoundedOrder α


instance PUnit.nonemptyFiniteLinearOrder : NonemptyFiniteLinearOrder PUnit where


instance Fin.nonemptyFiniteLinearOrder (n : ℕ) : NonemptyFiniteLinearOrder (Fin (n + 1)) where


instance ULift.nonemptyFiniteLinearOrder (α : Type u) [NonemptyFiniteLinearOrder α] :
    NonemptyFiniteLinearOrder (ULift.{v} α) :=
  { LinearOrder.lift' Equiv.ulift (Equiv.injective _) with }


instance (α : Type*) [NonemptyFiniteLinearOrder α] : NonemptyFiniteLinearOrder αᵒᵈ :=
  { OrderDual.fintype α with }


/-- The category of nonempty finite linear orders. -/
def NonemptyFinLinOrd :=
  Bundled NonemptyFiniteLinearOrder


instance : BundledHom.ParentProjection @NonemptyFiniteLinearOrder.toLinearOrder :=
  ⟨⟩


deriving instance LargeCategory for NonemptyFinLinOrd

-- Porting note: probably see https://github.com/leanprover-community/mathlib4/issues/5020

instance : ConcreteCategory NonemptyFinLinOrd :=
  BundledHom.concreteCategory _


instance : CoeSort NonemptyFinLinOrd Type* :=
  Bundled.coeSort


/-- Construct a bundled `NonemptyFinLinOrd` from the underlying type and typeclass. -/
def of (α : Type*) [NonemptyFiniteLinearOrder α] : NonemptyFinLinOrd :=
  Bundled.of α


@[simp]
theorem coe_of (α : Type*) [NonemptyFiniteLinearOrder α] : ↥(of α) = α :=
  rfl


instance : Inhabited NonemptyFinLinOrd :=
  ⟨of PUnit⟩


instance (α : NonemptyFinLinOrd) : NonemptyFiniteLinearOrder α :=
  α.str


instance hasForgetToLinOrd : HasForget₂ NonemptyFinLinOrd LinOrd :=
  BundledHom.forget₂ _ _


instance hasForgetToFinPartOrd : HasForget₂ NonemptyFinLinOrd FinPartOrd where
  forget₂ :=
    { obj := fun X => FinPartOrd.of X
      map := @fun _ _ => id }


/-- Constructs an equivalence between nonempty finite linear orders from an order isomorphism
between them. -/
@[simps]
def Iso.mk {α β : NonemptyFinLinOrd.{u}} (e : α ≃o β) : α ≅ β where
  hom := (e : OrderHom _ _)
  inv := (e.symm : OrderHom _ _)
  hom_inv_id := by
    /-
      α β : NonemptyFinLinOrd
      e : OrderIso ↑α ↑β
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ↑e ↑e.symm) (CategoryTheory.CategoryS …
    -/
    ext x
    /-
      case w
      α β : NonemptyFinLinOrd
      e : OrderIso ↑α ↑β
      x : (CategoryTheory.forget NonemptyFinLinOrd).obj α
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ↑e ↑e.symm) x) ((CategoryTheory.Cate …
    -/
    exact e.symm_apply_apply x
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      α β : NonemptyFinLinOrd
      e : OrderIso ↑α ↑β
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ↑e.symm ↑e) (CategoryTheory.CategoryS …
    -/
    ext x
    /-
      case w
      α β : NonemptyFinLinOrd
      e : OrderIso ↑α ↑β
      x : (CategoryTheory.forget NonemptyFinLinOrd).obj β
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ↑e.symm ↑e) x) ((CategoryTheory.Cate …
    -/
    exact e.apply_symm_apply x
    /-
      🎉 no goals
    -/


/-- `OrderDual` as a functor. -/
@[simps]
def dual : NonemptyFinLinOrd ⥤ NonemptyFinLinOrd where
  obj X := of Xᵒᵈ
  map := OrderHom.dual


/-- The equivalence between `NonemptyFinLinOrd` and itself induced by `OrderDual` both ways. -/
@[simps functor inverse]
def dualEquiv : NonemptyFinLinOrd ≌ NonemptyFinLinOrd where
  functor := dual
  inverse := dual
             /-
               ⊢ ∀ {X Y : NonemptyFinLinOrd} (f : Quiver.Hom X Y), Eq (CategoryTheory.Categor …
             -/
  unitIso := NatIso.ofComponents fun X => Iso.mk <| OrderIso.dualDual X
             /-
               🎉 no goals
             -/
               /-
                 ⊢ ∀ {X Y : NonemptyFinLinOrd} (f : Quiver.Hom X Y), Eq (CategoryTheory.Categor …
               -/
  counitIso := NatIso.ofComponents fun X => Iso.mk <| OrderIso.dualDual X
               /-
                 🎉 no goals
               -/


instance {A B : NonemptyFinLinOrd.{u}} : FunLike (A ⟶ B) A B where
  coe f := ⇑(show OrderHom A B from f)
  coe_injective' _ _ h := by
    /-
      A B : NonemptyFinLinOrd
      x✝¹ x✝ : Quiver.Hom A B
      h : Eq ((fun f => ⇑(letFun f fun this => this)) x✝¹) ((fun f => ⇑(letFun f fun …
      ⊢ Eq x✝¹ x✝
    -/
    ext x
    /-
      case w
      A B : NonemptyFinLinOrd
      x✝¹ x✝ : Quiver.Hom A B
      h : Eq ((fun f => ⇑(letFun f fun this => this)) x✝¹) ((fun f => ⇑(letFun f fun …
      x : (CategoryTheory.forget NonemptyFinLinOrd).obj A
      ⊢ Eq (x✝¹ x) (x✝ x)
    -/
    exact congr_fun h x
    /-
      🎉 no goals
    -/


instance {A B : NonemptyFinLinOrd.{u}} : OrderHomClass (A ⟶ B) A B where
  map_rel f _ _ h := f.monotone h


theorem mono_iff_injective {A B : NonemptyFinLinOrd.{u}} (f : A ⟶ B) :
    Mono f ↔ Function.Injective f := by
  /-
    A B : NonemptyFinLinOrd
    f : Quiver.Hom A B
    ⊢ Iff (CategoryTheory.Mono f) (Function.Injective ⇑f)
  -/
  refine ⟨?_, ConcreteCategory.mono_of_injective f⟩
  /-
    A B : NonemptyFinLinOrd
    f : Quiver.Hom A B
    ⊢ CategoryTheory.Mono f → Function.Injective ⇑f
  -/
  intro
  /-
    A B : NonemptyFinLinOrd
    f : Quiver.Hom A B
    a✝ : CategoryTheory.Mono f
    ⊢ Function.Injective ⇑f
  -/
  intro a₁ a₂ h
  /-
    A B : NonemptyFinLinOrd
    f : Quiver.Hom A B
    a✝ : CategoryTheory.Mono f
    a₁ a₂ : ↑A
    h : Eq (f a₁) (f a₂)
    ⊢ Eq a₁ a₂
  -/
  let X := NonemptyFinLinOrd.of (ULift (Fin 1))
  /-
    A B : NonemptyFinLinOrd
    f : Quiver.Hom A B
    a✝ : CategoryTheory.Mono f
    a₁ a₂ : ↑A
    h : Eq (f a₁) (f a₂)
    X : NonemptyFinLinOrd := NonemptyFinLinOrd.of (ULift.{?u.17570, 0} (Fin 1))
    ⊢ Eq a₁ a₂
  -/
  let g₁ : X ⟶ A := ⟨fun _ => a₁, fun _ _ _ => by rfl⟩
  /-
    A B : NonemptyFinLinOrd
    f : Quiver.Hom A B
    a✝ : CategoryTheory.Mono f
    a₁ a₂ : ↑A
    h : Eq (f a₁) (f a₂)
    X : NonemptyFinLinOrd := NonemptyFinLinOrd.of (ULift.{u, 0} (Fin 1))
    g₁ : Quiver.Hom X A := { toFun := fun x => a₁, monotone' := ⋯ }
    ⊢ Eq a₁ a₂
  -/
  let g₂ : X ⟶ A := ⟨fun _ => a₂, fun _ _ _ => by rfl⟩
  /-
    A B : NonemptyFinLinOrd
    f : Quiver.Hom A B
    a✝ : CategoryTheory.Mono f
    a₁ a₂ : ↑A
    h : Eq (f a₁) (f a₂)
    X : NonemptyFinLinOrd := NonemptyFinLinOrd.of (ULift.{u, 0} (Fin 1))
    g₁ : Quiver.Hom X A := { toFun := fun x => a₁, monotone' := ⋯ }
    g₂ : Quiver.Hom X A := { toFun := fun x => a₂, monotone' := ⋯ }
    ⊢ Eq a₁ a₂
  -/
  change g₁ (ULift.up (0 : Fin 1)) = g₂ (ULift.up (0 : Fin 1))
  have eq : g₁ ≫ f = g₂ ≫ f := by
    ext
    exact h
  /-
    A B : NonemptyFinLinOrd
    f : Quiver.Hom A B
    a✝ : CategoryTheory.Mono f
    a₁ a₂ : ↑A
    h : Eq (f a₁) (f a₂)
    X : NonemptyFinLinOrd := NonemptyFinLinOrd.of (ULift.{u, 0} (Fin 1))
    g₁ : Quiver.Hom X A := { toFun := fun x => a₁, monotone' := ⋯ }
    g₂ : Quiver.Hom X A := { toFun := fun x => a₂, monotone' := ⋯ }
    eq : Eq (CategoryTheory.CategoryStruct.comp g₁ f) (CategoryTheory.CategoryStru …
    ⊢ Eq (g₁ { down := 0 }) (g₂ { down := 0 })
  -/
  rw [cancel_mono] at eq
  /-
    A B : NonemptyFinLinOrd
    f : Quiver.Hom A B
    a✝ : CategoryTheory.Mono f
    a₁ a₂ : ↑A
    h : Eq (f a₁) (f a₂)
    X : NonemptyFinLinOrd := NonemptyFinLinOrd.of (ULift.{u, 0} (Fin 1))
    g₁ : Quiver.Hom X A := { toFun := fun x => a₁, monotone' := ⋯ }
    g₂ : Quiver.Hom X A := { toFun := fun x => a₂, monotone' := ⋯ }
    eq : Eq g₁ g₂
    ⊢ Eq (g₁ { down := 0 }) (g₂ { down := 0 })
  -/
  rw [eq]
  /-
    🎉 no goals
  -/

-- Porting note: added to ease the following proof

lemma forget_map_apply {A B : NonemptyFinLinOrd.{u}} (f : A ⟶ B) (a : A) :
    (forget NonemptyFinLinOrd).map f a = (f : OrderHom A B).toFun a := rfl


theorem epi_iff_surjective {A B : NonemptyFinLinOrd.{u}} (f : A ⟶ B) :
    Epi f ↔ Function.Surjective f := by
  /-
    A B : NonemptyFinLinOrd
    f : Quiver.Hom A B
    ⊢ Iff (CategoryTheory.Epi f) (Function.Surjective ⇑f)
  -/
  constructor
    /-
      case mp
      A B : NonemptyFinLinOrd
      f : Quiver.Hom A B
      ⊢ CategoryTheory.Epi f → Function.Surjective ⇑f
    -/
  · intro
    /-
      case mp
      A B : NonemptyFinLinOrd
      f : Quiver.Hom A B
      a✝ : CategoryTheory.Epi f
      ⊢ Function.Surjective ⇑f
    -/
    dsimp only [Function.Surjective]
    /-
      case mp
      A B : NonemptyFinLinOrd
      f : Quiver.Hom A B
      a✝ : CategoryTheory.Epi f
      ⊢ ∀ (b : ↑B), Exists fun a => Eq (f a) b
    -/
    by_contra! hf'
    /-
      case mp
      A B : NonemptyFinLinOrd
      f : Quiver.Hom A B
      a✝ : CategoryTheory.Epi f
      hf' : Exists fun b => ∀ (a : ↑A), Ne (f a) b
      ⊢ False
    -/
    rcases hf' with ⟨m, hm⟩
    /-
      case mp.intro
      A B : NonemptyFinLinOrd
      f : Quiver.Hom A B
      a✝ : CategoryTheory.Epi f
      m : ↑B
      hm : ∀ (a : ↑A), Ne (f a) m
      ⊢ False
    -/
    let Y := NonemptyFinLinOrd.of (ULift (Fin 2))
    let p₁ : B ⟶ Y :=
      ⟨fun b => if b < m then ULift.up 0 else ULift.up 1, fun x₁ x₂ h => by
        simp only
        split_ifs with h₁ h₂ h₂
        any_goals apply Fin.zero_le
        · exfalso
          exact h₁ (lt_of_le_of_lt h h₂)
        · rfl⟩
    let p₂ : B ⟶ Y :=
      ⟨fun b => if b ≤ m then ULift.up 0 else ULift.up 1, fun x₁ x₂ h => by
        simp only
        split_ifs with h₁ h₂ h₂
        any_goals apply Fin.zero_le
        · exfalso
          exact h₁ (h.trans h₂)
        · rfl⟩
    have h : p₁ m = p₂ m := by
      congr
      rw [← cancel_epi f]
      ext a
      simp only [coe_of, comp_apply]
      change ite _ _ _ = ite _ _ _
      split_ifs with h₁ h₂ h₂
      any_goals rfl
      · exfalso
        exact h₂ (le_of_lt h₁)
      · exfalso
        exact hm a (eq_of_le_of_not_lt h₂ h₁)
    /-
      case mp.intro
      A B : NonemptyFinLinOrd
      f : Quiver.Hom A B
      a✝ : CategoryTheory.Epi f
      m : ↑B
      hm : ∀ (a : ↑A), Ne (f a) m
      Y : NonemptyFinLinOrd := NonemptyFinLinOrd.of (ULift.{u, 0} (Fin 2))
      p₁ : Quiver.Hom B Y := { toFun := fun b => ite (LT.lt b m) { down := 0 } { dow …
      p₂ : Quiver.Hom B Y := { toFun := fun b => ite (LE.le b m) { down := 0 } { dow …
      h : Eq (p₁ m) (p₂ m)
      ⊢ False
    -/
    simp [Y, p₁, p₂, DFunLike.coe] at h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      A B : NonemptyFinLinOrd
      f : Quiver.Hom A B
      ⊢ Function.Surjective ⇑f → CategoryTheory.Epi f
    -/
  · intro h
    /-
      case mpr
      A B : NonemptyFinLinOrd
      f : Quiver.Hom A B
      h : Function.Surjective ⇑f
      ⊢ CategoryTheory.Epi f
    -/
    exact ConcreteCategory.epi_of_surjective f h
    /-
      🎉 no goals
    -/


instance : SplitEpiCategory NonemptyFinLinOrd.{u} :=
  ⟨fun {X Y} f hf => by
    have H : ∀ y : Y, Nonempty (f ⁻¹' {y}) := by
      rw [epi_iff_surjective] at hf
      intro y
      exact Nonempty.intro ⟨(hf y).choose, (hf y).choose_spec⟩
    /-
      X Y : NonemptyFinLinOrd
      f : Quiver.Hom X Y
      hf : CategoryTheory.Epi f
      H : ∀ (y : ↑Y), Nonempty ↑(Set.preimage (⇑f) (Singleton.singleton y))
      ⊢ CategoryTheory.IsSplitEpi f
    -/
    let φ : Y → X := fun y => (H y).some.1
    /-
      X Y : NonemptyFinLinOrd
      f : Quiver.Hom X Y
      hf : CategoryTheory.Epi f
      H : ∀ (y : ↑Y), Nonempty ↑(Set.preimage (⇑f) (Singleton.singleton y))
      φ : ↑Y → ↑X := fun y => ↑⋯.some
      ⊢ CategoryTheory.IsSplitEpi f
    -/
    have hφ : ∀ y : Y, f (φ y) = y := fun y => (H y).some.2
    /-
      X Y : NonemptyFinLinOrd
      f : Quiver.Hom X Y
      hf : CategoryTheory.Epi f
      H : ∀ (y : ↑Y), Nonempty ↑(Set.preimage (⇑f) (Singleton.singleton y))
      φ : ↑Y → ↑X := fun y => ↑⋯.some
      hφ : ∀ (y : ↑Y), Eq (f (φ y)) y
      ⊢ CategoryTheory.IsSplitEpi f
    -/
    refine IsSplitEpi.mk' ⟨⟨φ, ?_⟩, ?_⟩
    /-
      case refine_1
      X Y : NonemptyFinLinOrd
      f : Quiver.Hom X Y
      hf : CategoryTheory.Epi f
      H : ∀ (y : ↑Y), Nonempty ↑(Set.preimage (⇑f) (Singleton.singleton y))
      φ : ↑Y → ↑X := fun y => ↑⋯.some
      hφ : ∀ (y : ↑Y), Eq (f (φ y)) y
      ⊢ Monotone φ
    -/
    swap
      /-
        case refine_2
        X Y : NonemptyFinLinOrd
        f : Quiver.Hom X Y
        hf : CategoryTheory.Epi f
        H : ∀ (y : ↑Y), Nonempty ↑(Set.preimage (⇑f) (Singleton.singleton y))
        φ : ↑Y → ↑X := fun y => ↑⋯.some
        hφ : ∀ (y : ↑Y), Eq (f (φ y)) y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := φ, monotone' := ?refine_1  …
      -/
    · ext b
      /-
        case refine_2.w
        X Y : NonemptyFinLinOrd
        f : Quiver.Hom X Y
        hf : CategoryTheory.Epi f
        H : ∀ (y : ↑Y), Nonempty ↑(Set.preimage (⇑f) (Singleton.singleton y))
        φ : ↑Y → ↑X := fun y => ↑⋯.some
        hφ : ∀ (y : ↑Y), Eq (f (φ y)) y
        b : (CategoryTheory.forget NonemptyFinLinOrd).obj Y
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp { toFun := φ, monotone' := ?refine_1 …
      -/
      apply hφ
      /-
        🎉 no goals
      -/
      /-
        case refine_1
        X Y : NonemptyFinLinOrd
        f : Quiver.Hom X Y
        hf : CategoryTheory.Epi f
        H : ∀ (y : ↑Y), Nonempty ↑(Set.preimage (⇑f) (Singleton.singleton y))
        φ : ↑Y → ↑X := fun y => ↑⋯.some
        hφ : ∀ (y : ↑Y), Eq (f (φ y)) y
        ⊢ Monotone φ
      -/
    · intro a b
      /-
        case refine_1
        X Y : NonemptyFinLinOrd
        f : Quiver.Hom X Y
        hf : CategoryTheory.Epi f
        H : ∀ (y : ↑Y), Nonempty ↑(Set.preimage (⇑f) (Singleton.singleton y))
        φ : ↑Y → ↑X := fun y => ↑⋯.some
        hφ : ∀ (y : ↑Y), Eq (f (φ y)) y
        a b : ↑Y
        ⊢ LE.le a b → LE.le (φ a) (φ b)
      -/
      contrapose
      /-
        case refine_1
        X Y : NonemptyFinLinOrd
        f : Quiver.Hom X Y
        hf : CategoryTheory.Epi f
        H : ∀ (y : ↑Y), Nonempty ↑(Set.preimage (⇑f) (Singleton.singleton y))
        φ : ↑Y → ↑X := fun y => ↑⋯.some
        hφ : ∀ (y : ↑Y), Eq (f (φ y)) y
        a b : ↑Y
        ⊢ Not (LE.le (φ a) (φ b)) → Not (LE.le a b)
      -/
      intro h
      /-
        case refine_1
        X Y : NonemptyFinLinOrd
        f : Quiver.Hom X Y
        hf : CategoryTheory.Epi f
        H : ∀ (y : ↑Y), Nonempty ↑(Set.preimage (⇑f) (Singleton.singleton y))
        φ : ↑Y → ↑X := fun y => ↑⋯.some
        hφ : ∀ (y : ↑Y), Eq (f (φ y)) y
        a b : ↑Y
        h : Not (LE.le (φ a) (φ b))
        ⊢ Not (LE.le a b)
      -/
      simp only [not_le] at h ⊢
      suffices b ≤ a by
        apply lt_of_le_of_ne this
        rintro rfl
        exfalso
        simp at h
      /-
        case refine_1
        X Y : NonemptyFinLinOrd
        f : Quiver.Hom X Y
        hf : CategoryTheory.Epi f
        H : ∀ (y : ↑Y), Nonempty ↑(Set.preimage (⇑f) (Singleton.singleton y))
        φ : ↑Y → ↑X := fun y => ↑⋯.some
        hφ : ∀ (y : ↑Y), Eq (f (φ y)) y
        a b : ↑Y
        h : LT.lt (φ b) (φ a)
        ⊢ LE.le b a
      -/
      have H : f (φ b) ≤ f (φ a) := f.monotone (le_of_lt h)
      /-
        case refine_1
        X Y : NonemptyFinLinOrd
        f : Quiver.Hom X Y
        hf : CategoryTheory.Epi f
        H✝ : ∀ (y : ↑Y), Nonempty ↑(Set.preimage (⇑f) (Singleton.singleton y))
        φ : ↑Y → ↑X := fun y => ↑⋯.some
        hφ : ∀ (y : ↑Y), Eq (f (φ y)) y
        a b : ↑Y
        h : LT.lt (φ b) (φ a)
        H : LE.le (f (φ b)) (f (φ a))
        ⊢ LE.le b a
      -/
      simpa only [hφ] using H⟩
      /-
        🎉 no goals
      -/


instance : HasStrongEpiMonoFactorisations NonemptyFinLinOrd.{u} :=
  ⟨fun {X Y} f => by
    /-
      X Y : NonemptyFinLinOrd
      f : Quiver.Hom X Y
      ⊢ Nonempty (CategoryTheory.Limits.StrongEpiMonoFactorisation f)
    -/
    letI : NonemptyFiniteLinearOrder (Set.image f ⊤) := ⟨by infer_instance⟩
    /-
      X Y : NonemptyFinLinOrd
      f : Quiver.Hom X Y
      this : NonemptyFiniteLinearOrder ↑(Set.image (⇑f) Top.top) := NonemptyFiniteLi …
      ⊢ Nonempty (CategoryTheory.Limits.StrongEpiMonoFactorisation f)
    -/
    let I := NonemptyFinLinOrd.of (Set.image f ⊤)
    /-
      X Y : NonemptyFinLinOrd
      f : Quiver.Hom X Y
      this : NonemptyFiniteLinearOrder ↑(Set.image (⇑f) Top.top) := NonemptyFiniteLi …
      I : NonemptyFinLinOrd := NonemptyFinLinOrd.of ↑(Set.image (⇑f) Top.top)
      ⊢ Nonempty (CategoryTheory.Limits.StrongEpiMonoFactorisation f)
    -/
    let e : X ⟶ I := ⟨fun x => ⟨f x, ⟨x, by tauto⟩⟩, fun x₁ x₂ h => f.monotone h⟩
    /-
      X Y : NonemptyFinLinOrd
      f : Quiver.Hom X Y
      this : NonemptyFiniteLinearOrder ↑(Set.image (⇑f) Top.top) := NonemptyFiniteLi …
      I : NonemptyFinLinOrd := NonemptyFinLinOrd.of ↑(Set.image (⇑f) Top.top)
      e : Quiver.Hom X I := { toFun := fun x => ⟨f x, ⋯⟩, monotone' := ⋯ }
      ⊢ Nonempty (CategoryTheory.Limits.StrongEpiMonoFactorisation f)
    -/
    let m : I ⟶ Y := ⟨fun y => y.1, by tauto⟩
    haveI : Epi e := by
      rw [epi_iff_surjective]
      rintro ⟨_, y, h, rfl⟩
      exact ⟨y, rfl⟩
    /-
      X Y : NonemptyFinLinOrd
      f : Quiver.Hom X Y
      this✝ : NonemptyFiniteLinearOrder ↑(Set.image (⇑f) Top.top) := NonemptyFiniteL …
      I : NonemptyFinLinOrd := NonemptyFinLinOrd.of ↑(Set.image (⇑f) Top.top)
      e : Quiver.Hom X I := { toFun := fun x => ⟨f x, ⋯⟩, monotone' := ⋯ }
      m : Quiver.Hom I Y := { toFun := fun y => ↑y, monotone' := ⋯ }
      this : CategoryTheory.Epi e
      ⊢ Nonempty (CategoryTheory.Limits.StrongEpiMonoFactorisation f)
    -/
    haveI : StrongEpi e := strongEpi_of_epi e
    /-
      X Y : NonemptyFinLinOrd
      f : Quiver.Hom X Y
      this✝¹ : NonemptyFiniteLinearOrder ↑(Set.image (⇑f) Top.top) := NonemptyFinite …
      I : NonemptyFinLinOrd := NonemptyFinLinOrd.of ↑(Set.image (⇑f) Top.top)
      e : Quiver.Hom X I := { toFun := fun x => ⟨f x, ⋯⟩, monotone' := ⋯ }
      m : Quiver.Hom I Y := { toFun := fun y => ↑y, monotone' := ⋯ }
      this✝ : CategoryTheory.Epi e
      this : CategoryTheory.StrongEpi e
      ⊢ Nonempty (CategoryTheory.Limits.StrongEpiMonoFactorisation f)
    -/
    haveI : Mono m := ConcreteCategory.mono_of_injective _ (fun x y h => Subtype.ext h)
    /-
      X Y : NonemptyFinLinOrd
      f : Quiver.Hom X Y
      this✝² : NonemptyFiniteLinearOrder ↑(Set.image (⇑f) Top.top) := NonemptyFinite …
      I : NonemptyFinLinOrd := NonemptyFinLinOrd.of ↑(Set.image (⇑f) Top.top)
      e : Quiver.Hom X I := { toFun := fun x => ⟨f x, ⋯⟩, monotone' := ⋯ }
      m : Quiver.Hom I Y := { toFun := fun y => ↑y, monotone' := ⋯ }
      this✝¹ : CategoryTheory.Epi e
      this✝ : CategoryTheory.StrongEpi e
      this : CategoryTheory.Mono m
      ⊢ Nonempty (CategoryTheory.Limits.StrongEpiMonoFactorisation f)
    -/
    exact ⟨⟨I, m, e, rfl⟩⟩⟩
    /-
      🎉 no goals
    -/


theorem nonemptyFinLinOrd_dual_comp_forget_to_linOrd :
    NonemptyFinLinOrd.dual ⋙ forget₂ NonemptyFinLinOrd LinOrd =
      forget₂ NonemptyFinLinOrd LinOrd ⋙ LinOrd.dual :=
  rfl


/-- The forgetful functor `NonemptyFinLinOrd ⥤ FinPartOrd` and `OrderDual` commute. -/
def nonemptyFinLinOrdDualCompForgetToFinPartOrd :
    NonemptyFinLinOrd.dual ⋙ forget₂ NonemptyFinLinOrd FinPartOrd ≅
      forget₂ NonemptyFinLinOrd FinPartOrd ⋙ FinPartOrd.dual where
  hom := { app := fun X => OrderHom.id }
  inv := { app := fun X => OrderHom.id }


/-- The generating arrow `i ⟶ i+1` in the category `Fin n`.-/
def Fin.hom_succ {n} (i : Fin n) : i.castSucc ⟶ i.succ := homOfLE (Fin.castSucc_le_succ i)

