/-- If `F : Cᵒᵖ ⥤ D` is a presheaf with values in a concrete category, if `x` and `y` are
elements in `F.obj X`, this is the sieve of `X.unop` consisting of morphisms `f`
such that `F.map f.op x = F.map f.op y`. -/
@[simps]
def equalizerSieve {F : Cᵒᵖ ⥤ D} {X : Cᵒᵖ} (x y : F.obj X) : Sieve X.unop where
  arrows _ f := F.map f.op x = F.map f.op y
  downward_closed {X Y} f hf g := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F : CategoryTheory.Functor (Opposite C) D
      X✝ : Opposite C
      x y : (CategoryTheory.forget D).obj (F.obj X✝)
      X Y : C
      f : Quiver.Hom X (Opposite.unop X✝)
      hf : (fun x_1 f => Eq ((F.map f.op) x) ((F.map f.op) y)) X f
      g : Quiver.Hom Y X
      ⊢ (fun x_1 f => Eq ((F.map f.op) x) ((F.map f.op) y)) Y (CategoryTheory.Catego …
    -/
    dsimp at hf ⊢
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F : CategoryTheory.Functor (Opposite C) D
      X✝ : Opposite C
      x y : (CategoryTheory.forget D).obj (F.obj X✝)
      X Y : C
      f : Quiver.Hom X (Opposite.unop X✝)
      hf : Eq ((F.map f.op) x) ((F.map f.op) y)
      g : Quiver.Hom Y X
      ⊢ Eq ((F.map (CategoryTheory.CategoryStruct.comp f.op g.op)) x) ((F.map (Categ …
    -/
    simp [hf]
    /-
      🎉 no goals
    -/


@[simp]
lemma equalizerSieve_self_eq_top {F : Cᵒᵖ ⥤ D} {X : Cᵒᵖ} (x : F.obj X) :
                                 /-
                                   C : Type u
                                   inst✝² : CategoryTheory.Category.{v, u} C
                                   D : Type u'
                                   inst✝¹ : CategoryTheory.Category.{v', u'} D
                                   inst✝ : CategoryTheory.ConcreteCategory D
                                   F : CategoryTheory.Functor (Opposite C) D
                                   X : Opposite C
                                   x : (CategoryTheory.forget D).obj (F.obj X)
                                   ⊢ Eq (CategoryTheory.Presheaf.equalizerSieve x x) Top.top
                                 -/
    equalizerSieve x x = ⊤ := by aesop
                                 /-
                                   🎉 no goals
                                 -/


@[simp]
lemma equalizerSieve_eq_top_iff {F : Cᵒᵖ ⥤ D} {X : Cᵒᵖ} (x y : F.obj X) :
    equalizerSieve x y = ⊤ ↔ x = y := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} D
    inst✝ : CategoryTheory.ConcreteCategory D
    F : CategoryTheory.Functor (Opposite C) D
    X : Opposite C
    x y : (CategoryTheory.forget D).obj (F.obj X)
    ⊢ Iff (Eq (CategoryTheory.Presheaf.equalizerSieve x y) Top.top) (Eq x y)
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.ConcreteCategory D
      F : CategoryTheory.Functor (Opposite C) D
      X : Opposite C
      x y : (CategoryTheory.forget D).obj (F.obj X)
      ⊢ Eq (CategoryTheory.Presheaf.equalizerSieve x y) Top.top → Eq x y
    -/
  · intro h
    /-
      case mp
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.ConcreteCategory D
      F : CategoryTheory.Functor (Opposite C) D
      X : Opposite C
      x y : (CategoryTheory.forget D).obj (F.obj X)
      h : Eq (CategoryTheory.Presheaf.equalizerSieve x y) Top.top
      ⊢ Eq x y
    -/
    simpa using (show equalizerSieve x y (𝟙 _) by simp [h])
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.ConcreteCategory D
      F : CategoryTheory.Functor (Opposite C) D
      X : Opposite C
      x y : (CategoryTheory.forget D).obj (F.obj X)
      ⊢ Eq x y → Eq (CategoryTheory.Presheaf.equalizerSieve x y) Top.top
    -/
  · rintro rfl
    /-
      case mpr
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.ConcreteCategory D
      F : CategoryTheory.Functor (Opposite C) D
      X : Opposite C
      x : (CategoryTheory.forget D).obj (F.obj X)
      ⊢ Eq (CategoryTheory.Presheaf.equalizerSieve x x) Top.top
    -/
    apply equalizerSieve_self_eq_top
    /-
      🎉 no goals
    -/


/-- A morphism `φ : F₁ ⟶ F₂` of presheaves `Cᵒᵖ ⥤ D` (with `D` a concrete category)
is locally injective for a Grothendieck topology `J` on `C` if
whenever two sections of `F₁` are sent to the same section of `F₂`, then these two
sections coincide locally. -/
class IsLocallyInjective : Prop where
  equalizerSieve_mem {X : Cᵒᵖ} (x y : F₁.obj X) (h : φ.app X x = φ.app X y) :
    equalizerSieve x y ∈ J X.unop


lemma equalizerSieve_mem [IsLocallyInjective J φ]
    {X : Cᵒᵖ} (x y : F₁.obj X) (h : φ.app X x = φ.app X y) :
    equalizerSieve x y ∈ J X.unop :=
  IsLocallyInjective.equalizerSieve_mem x y h


lemma isLocallyInjective_of_injective (hφ : ∀ (X : Cᵒᵖ), Function.Injective (φ.app X)) :
    IsLocallyInjective J φ where
  equalizerSieve_mem {X} x y h := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      hφ : ∀ (X : Opposite C), Function.Injective ⇑(φ.app X)
      X : Opposite C
      x y : (CategoryTheory.forget D).obj (F₁.obj X)
      h : Eq ((φ.app X) x) ((φ.app X) y)
      ⊢ Membership.mem (J (Opposite.unop X)) (CategoryTheory.Presheaf.equalizerSieve …
    -/
    convert J.top_mem X.unop
    /-
      case h.e'_5
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      hφ : ∀ (X : Opposite C), Function.Injective ⇑(φ.app X)
      X : Opposite C
      x y : (CategoryTheory.forget D).obj (F₁.obj X)
      h : Eq ((φ.app X) x) ((φ.app X) y)
      ⊢ Eq (CategoryTheory.Presheaf.equalizerSieve x y) Top.top
    -/
    ext Y f
    /-
      case h.e'_5.h
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      hφ : ∀ (X : Opposite C), Function.Injective ⇑(φ.app X)
      X : Opposite C
      x y : (CategoryTheory.forget D).obj (F₁.obj X)
      h : Eq ((φ.app X) x) ((φ.app X) y)
      Y : C
      f : Quiver.Hom Y (Opposite.unop X)
      ⊢ Iff ((CategoryTheory.Presheaf.equalizerSieve x y).arrows f) (Top.top.arrows f)
    -/
    simp only [equalizerSieve_apply, op_unop, Sieve.top_apply, iff_true]
    /-
      case h.e'_5.h
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      hφ : ∀ (X : Opposite C), Function.Injective ⇑(φ.app X)
      X : Opposite C
      x y : (CategoryTheory.forget D).obj (F₁.obj X)
      h : Eq ((φ.app X) x) ((φ.app X) y)
      Y : C
      f : Quiver.Hom Y (Opposite.unop X)
      ⊢ Eq ((F₁.map f.op) x) ((F₁.map f.op) y)
    -/
    apply hφ
    /-
      case h.e'_5.h.a
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      hφ : ∀ (X : Opposite C), Function.Injective ⇑(φ.app X)
      X : Opposite C
      x y : (CategoryTheory.forget D).obj (F₁.obj X)
      h : Eq ((φ.app X) x) ((φ.app X) y)
      Y : C
      f : Quiver.Hom Y (Opposite.unop X)
      ⊢ Eq ((φ.app { unop := Y }) ((F₁.map f.op) x)) ((φ.app { unop := Y }) ((F₁.map …
    -/
    simp [h]
    /-
      🎉 no goals
    -/


instance [IsIso φ] : IsLocallyInjective J φ :=
  isLocallyInjective_of_injective J φ (fun X => Function.Bijective.injective (by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      inst✝¹ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      ψ : Quiver.Hom F₂ F₃
      inst✝ : CategoryTheory.IsIso φ
      X : Opposite C
      ⊢ Function.Bijective ⇑(φ.app X)
    -/
    rw [← isIso_iff_bijective]
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      inst✝¹ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      ψ : Quiver.Hom F₂ F₃
      inst✝ : CategoryTheory.IsIso φ
      X : Opposite C
      ⊢ CategoryTheory.IsIso ⇑(φ.app X)
    -/
    change IsIso ((forget D).map (φ.app X))
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      inst✝¹ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      ψ : Quiver.Hom F₂ F₃
      inst✝ : CategoryTheory.IsIso φ
      X : Opposite C
      ⊢ CategoryTheory.IsIso ((CategoryTheory.forget D).map (φ.app X))
    -/
    infer_instance))
    /-
      🎉 no goals
    -/


instance isLocallyInjective_forget [IsLocallyInjective J φ] :
    IsLocallyInjective J (whiskerRight φ (forget D)) where
  equalizerSieve_mem x y h := equalizerSieve_mem J φ x y h


lemma isLocallyInjective_forget_iff :
    IsLocallyInjective J (whiskerRight φ (forget D)) ↔ IsLocallyInjective J φ := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} D
    inst✝ : CategoryTheory.ConcreteCategory D
    J : CategoryTheory.GrothendieckTopology C
    F₁ F₂ : CategoryTheory.Functor (Opposite C) D
    φ : Quiver.Hom F₁ F₂
    ⊢ Iff (CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.whiskerRig …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      ⊢ CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.whiskerRight φ  …
    -/
  · intro
    /-
      case mp
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      a✝ : CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.whiskerRight …
      ⊢ CategoryTheory.Presheaf.IsLocallyInjective J φ
    -/
    exact ⟨fun x y h => equalizerSieve_mem J (whiskerRight φ (forget D)) x y h⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      ⊢ CategoryTheory.Presheaf.IsLocallyInjective J φ → CategoryTheory.Presheaf.IsL …
    -/
  · intro
    /-
      case mpr
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      a✝ : CategoryTheory.Presheaf.IsLocallyInjective J φ
      ⊢ CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.whiskerRight φ  …
    -/
    infer_instance
    /-
      🎉 no goals
    -/


lemma isLocallyInjective_iff_equalizerSieve_mem_imp :
    IsLocallyInjective J φ ↔ ∀ ⦃X : Cᵒᵖ⦄ (x y : F₁.obj X),
      equalizerSieve (φ.app _ x) (φ.app _ y) ∈ J X.unop → equalizerSieve x y ∈ J X.unop := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} D
    inst✝ : CategoryTheory.ConcreteCategory D
    J : CategoryTheory.GrothendieckTopology C
    F₁ F₂ : CategoryTheory.Functor (Opposite C) D
    φ : Quiver.Hom F₁ F₂
    ⊢ Iff (CategoryTheory.Presheaf.IsLocallyInjective J φ) (∀ ⦃X : Opposite C⦄ (x  …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      ⊢ CategoryTheory.Presheaf.IsLocallyInjective J φ → ∀ ⦃X : Opposite C⦄ (x y : ( …
    -/
  · intro _ X x y h
    /-
      case mp
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      a✝ : CategoryTheory.Presheaf.IsLocallyInjective J φ
      X : Opposite C
      x y : (CategoryTheory.forget D).obj (F₁.obj X)
      h : Membership.mem (J (Opposite.unop X)) (CategoryTheory.Presheaf.equalizerSie …
      ⊢ Membership.mem (J (Opposite.unop X)) (CategoryTheory.Presheaf.equalizerSieve …
    -/
    let S := equalizerSieve (φ.app _ x) (φ.app _ y)
    let T : ∀ ⦃Y : C⦄ ⦃f : Y ⟶ X.unop⦄ (_ : S f), Sieve Y := fun Y f _ =>
      equalizerSieve (F₁.map f.op x) ((F₁.map f.op y))
    /-
      case mp
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      a✝ : CategoryTheory.Presheaf.IsLocallyInjective J φ
      X : Opposite C
      x y : (CategoryTheory.forget D).obj (F₁.obj X)
      h : Membership.mem (J (Opposite.unop X)) (CategoryTheory.Presheaf.equalizerSie …
      S : CategoryTheory.Sieve (Opposite.unop X) := CategoryTheory.Presheaf.equalize …
      T : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y (Opposite.unop X)⦄ → S.arrows f → CategoryTheo …
      ⊢ Membership.mem (J (Opposite.unop X)) (CategoryTheory.Presheaf.equalizerSieve …
    -/
    refine J.superset_covering ?_ (J.transitive h (Sieve.bind S.1 T) ?_)
      /-
        case mp.refine_1
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝¹ : CategoryTheory.Category.{v', u'} D
        inst✝ : CategoryTheory.ConcreteCategory D
        J : CategoryTheory.GrothendieckTopology C
        F₁ F₂ : CategoryTheory.Functor (Opposite C) D
        φ : Quiver.Hom F₁ F₂
        a✝ : CategoryTheory.Presheaf.IsLocallyInjective J φ
        X : Opposite C
        x y : (CategoryTheory.forget D).obj (F₁.obj X)
        h : Membership.mem (J (Opposite.unop X)) (CategoryTheory.Presheaf.equalizerSie …
        S : CategoryTheory.Sieve (Opposite.unop X) := CategoryTheory.Presheaf.equalize …
        T : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y (Opposite.unop X)⦄ → S.arrows f → CategoryTheo …
        ⊢ LE.le (CategoryTheory.Sieve.bind S.arrows T) (CategoryTheory.Presheaf.equali …
      -/
    · rintro Y f ⟨Z, a, g, hg, ha, rfl⟩
      /-
        case mp.refine_1.intro.intro.intro.intro.intro
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝¹ : CategoryTheory.Category.{v', u'} D
        inst✝ : CategoryTheory.ConcreteCategory D
        J : CategoryTheory.GrothendieckTopology C
        F₁ F₂ : CategoryTheory.Functor (Opposite C) D
        φ : Quiver.Hom F₁ F₂
        a✝ : CategoryTheory.Presheaf.IsLocallyInjective J φ
        X : Opposite C
        x y : (CategoryTheory.forget D).obj (F₁.obj X)
        h : Membership.mem (J (Opposite.unop X)) (CategoryTheory.Presheaf.equalizerSie …
        S : CategoryTheory.Sieve (Opposite.unop X) := CategoryTheory.Presheaf.equalize …
        T : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y (Opposite.unop X)⦄ → S.arrows f → CategoryTheo …
        Y Z : C
        a : Quiver.Hom Y Z
        g : Quiver.Hom Z (Opposite.unop X)
        hg : S.arrows g
        ha : (T hg).arrows a
        ⊢ (CategoryTheory.Presheaf.equalizerSieve x y).arrows (CategoryTheory.Category …
      -/
      simpa using ha
      /-
        🎉 no goals
      -/
      /-
        case mp.refine_2
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝¹ : CategoryTheory.Category.{v', u'} D
        inst✝ : CategoryTheory.ConcreteCategory D
        J : CategoryTheory.GrothendieckTopology C
        F₁ F₂ : CategoryTheory.Functor (Opposite C) D
        φ : Quiver.Hom F₁ F₂
        a✝ : CategoryTheory.Presheaf.IsLocallyInjective J φ
        X : Opposite C
        x y : (CategoryTheory.forget D).obj (F₁.obj X)
        h : Membership.mem (J (Opposite.unop X)) (CategoryTheory.Presheaf.equalizerSie …
        S : CategoryTheory.Sieve (Opposite.unop X) := CategoryTheory.Presheaf.equalize …
        T : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y (Opposite.unop X)⦄ → S.arrows f → CategoryTheo …
        ⊢ ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y (Opposite.unop X)⦄, (CategoryTheory.Presheaf.equ …
      -/
    · intro Y f hf
      refine J.superset_covering (Sieve.le_pullback_bind S.1 T _ hf)
        (equalizerSieve_mem J φ _ _ ?_)
      /-
        case mp.refine_2
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝¹ : CategoryTheory.Category.{v', u'} D
        inst✝ : CategoryTheory.ConcreteCategory D
        J : CategoryTheory.GrothendieckTopology C
        F₁ F₂ : CategoryTheory.Functor (Opposite C) D
        φ : Quiver.Hom F₁ F₂
        a✝ : CategoryTheory.Presheaf.IsLocallyInjective J φ
        X : Opposite C
        x y : (CategoryTheory.forget D).obj (F₁.obj X)
        h : Membership.mem (J (Opposite.unop X)) (CategoryTheory.Presheaf.equalizerSie …
        S : CategoryTheory.Sieve (Opposite.unop X) := CategoryTheory.Presheaf.equalize …
        T : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y (Opposite.unop X)⦄ → S.arrows f → CategoryTheo …
        Y : C
        f : Quiver.Hom Y (Opposite.unop X)
        hf : (CategoryTheory.Presheaf.equalizerSieve ((φ.app X) x) ((φ.app X) y)).arro …
        ⊢ Eq ((φ.app { unop := Y }) ((F₁.map f.op) x)) ((φ.app { unop := Y }) ((F₁.map …
      -/
      rw [NatTrans.naturality_apply, NatTrans.naturality_apply]
      /-
        case mp.refine_2
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝¹ : CategoryTheory.Category.{v', u'} D
        inst✝ : CategoryTheory.ConcreteCategory D
        J : CategoryTheory.GrothendieckTopology C
        F₁ F₂ : CategoryTheory.Functor (Opposite C) D
        φ : Quiver.Hom F₁ F₂
        a✝ : CategoryTheory.Presheaf.IsLocallyInjective J φ
        X : Opposite C
        x y : (CategoryTheory.forget D).obj (F₁.obj X)
        h : Membership.mem (J (Opposite.unop X)) (CategoryTheory.Presheaf.equalizerSie …
        S : CategoryTheory.Sieve (Opposite.unop X) := CategoryTheory.Presheaf.equalize …
        T : ⦃Y : C⦄ → ⦃f : Quiver.Hom Y (Opposite.unop X)⦄ → S.arrows f → CategoryTheo …
        Y : C
        f : Quiver.Hom Y (Opposite.unop X)
        hf : (CategoryTheory.Presheaf.equalizerSieve ((φ.app X) x) ((φ.app X) y)).arro …
        ⊢ Eq ((F₂.map f.op) ((φ.app { unop := Opposite.unop X }) x)) ((F₂.map f.op) (( …
      -/
      exact hf
      /-
        🎉 no goals
      -/
    /-
      case mpr
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      ⊢ (∀ ⦃X : Opposite C⦄ (x y : (CategoryTheory.forget D).obj (F₁.obj X)), Member …
    -/
  · intro hφ
    /-
      case mpr
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      hφ : ∀ ⦃X : Opposite C⦄ (x y : (CategoryTheory.forget D).obj (F₁.obj X)), Memb …
      ⊢ CategoryTheory.Presheaf.IsLocallyInjective J φ
    -/
    exact ⟨fun {X} x y h => hφ x y (by simp [h])⟩
    /-
      🎉 no goals
    -/


lemma equalizerSieve_mem_of_equalizerSieve_app_mem
    {X : Cᵒᵖ} (x y : F₁.obj X) (h : equalizerSieve (φ.app _ x) (φ.app _ y) ∈ J X.unop)
    [IsLocallyInjective J φ] :
    equalizerSieve x y ∈ J X.unop :=
  (isLocallyInjective_iff_equalizerSieve_mem_imp J φ).1 inferInstance x y h


instance isLocallyInjective_comp [IsLocallyInjective J φ] [IsLocallyInjective J ψ] :
    IsLocallyInjective J (φ ≫ ψ) where
  equalizerSieve_mem {X} x y h := by
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} D
      inst✝² : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      ψ : Quiver.Hom F₂ F₃
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
      inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J ψ
      X : Opposite C
      x y : (CategoryTheory.forget D).obj (F₁.obj X)
      h : Eq (((CategoryTheory.CategoryStruct.comp φ ψ).app X) x) (((CategoryTheory. …
      ⊢ Membership.mem (J (Opposite.unop X)) (CategoryTheory.Presheaf.equalizerSieve …
    -/
    apply equalizerSieve_mem_of_equalizerSieve_app_mem J φ
    /-
      case h
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} D
      inst✝² : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      ψ : Quiver.Hom F₂ F₃
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
      inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J ψ
      X : Opposite C
      x y : (CategoryTheory.forget D).obj (F₁.obj X)
      h : Eq (((CategoryTheory.CategoryStruct.comp φ ψ).app X) x) (((CategoryTheory. …
      ⊢ Membership.mem (J (Opposite.unop X)) (CategoryTheory.Presheaf.equalizerSieve …
    -/
    exact equalizerSieve_mem J ψ _ _ (by simpa using h)
    /-
      🎉 no goals
    -/


lemma isLocallyInjective_of_isLocallyInjective [IsLocallyInjective J (φ ≫ ψ)] :
    IsLocallyInjective J φ where
                                                                       /-
                                                                         C : Type u
                                                                         inst✝³ : CategoryTheory.Category.{v, u} C
                                                                         D : Type u'
                                                                         inst✝² : CategoryTheory.Category.{v', u'} D
                                                                         inst✝¹ : CategoryTheory.ConcreteCategory D
                                                                         J : CategoryTheory.GrothendieckTopology C
                                                                         F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) D
                                                                         φ : Quiver.Hom F₁ F₂
                                                                         ψ : Quiver.Hom F₂ F₃
                                                                         inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.CategoryS …
                                                                         X : Opposite C
                                                                         x y : (CategoryTheory.forget D).obj (F₁.obj X)
                                                                         h : Eq ((φ.app X) x) ((φ.app X) y)
                                                                         ⊢ Eq (((CategoryTheory.CategoryStruct.comp φ ψ).app X) x) (((CategoryTheory.Ca …
                                                                       -/
  equalizerSieve_mem {X} x y h := equalizerSieve_mem J (φ ≫ ψ) x y (by simp [h])
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


lemma isLocallyInjective_of_isLocallyInjective_fac {φψ : F₁ ⟶ F₃} (fac : φ ≫ ψ = φψ)
    [IsLocallyInjective J φψ] : IsLocallyInjective J φ := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} D
    inst✝¹ : CategoryTheory.ConcreteCategory D
    J : CategoryTheory.GrothendieckTopology C
    F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) D
    φ : Quiver.Hom F₁ F₂
    ψ : Quiver.Hom F₂ F₃
    φψ : Quiver.Hom F₁ F₃
    fac : Eq (CategoryTheory.CategoryStruct.comp φ ψ) φψ
    inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J φψ
    ⊢ CategoryTheory.Presheaf.IsLocallyInjective J φ
  -/
  subst fac
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} D
    inst✝¹ : CategoryTheory.ConcreteCategory D
    J : CategoryTheory.GrothendieckTopology C
    F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) D
    φ : Quiver.Hom F₁ F₂
    ψ : Quiver.Hom F₂ F₃
    inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.CategoryS …
    ⊢ CategoryTheory.Presheaf.IsLocallyInjective J φ
  -/
  exact isLocallyInjective_of_isLocallyInjective J φ ψ
  /-
    🎉 no goals
  -/


lemma isLocallyInjective_iff_of_fac {φψ : F₁ ⟶ F₃} (fac : φ ≫ ψ = φψ) [IsLocallyInjective J ψ] :
    IsLocallyInjective J φψ ↔ IsLocallyInjective J φ := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} D
    inst✝¹ : CategoryTheory.ConcreteCategory D
    J : CategoryTheory.GrothendieckTopology C
    F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) D
    φ : Quiver.Hom F₁ F₂
    ψ : Quiver.Hom F₂ F₃
    φψ : Quiver.Hom F₁ F₃
    fac : Eq (CategoryTheory.CategoryStruct.comp φ ψ) φψ
    inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J ψ
    ⊢ Iff (CategoryTheory.Presheaf.IsLocallyInjective J φψ) (CategoryTheory.Preshe …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      inst✝¹ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      ψ : Quiver.Hom F₂ F₃
      φψ : Quiver.Hom F₁ F₃
      fac : Eq (CategoryTheory.CategoryStruct.comp φ ψ) φψ
      inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J ψ
      ⊢ CategoryTheory.Presheaf.IsLocallyInjective J φψ → CategoryTheory.Presheaf.Is …
    -/
  · intro
    /-
      case mp
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      inst✝¹ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      ψ : Quiver.Hom F₂ F₃
      φψ : Quiver.Hom F₁ F₃
      fac : Eq (CategoryTheory.CategoryStruct.comp φ ψ) φψ
      inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J ψ
      a✝ : CategoryTheory.Presheaf.IsLocallyInjective J φψ
      ⊢ CategoryTheory.Presheaf.IsLocallyInjective J φ
    -/
    exact isLocallyInjective_of_isLocallyInjective_fac J fac
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      inst✝¹ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      ψ : Quiver.Hom F₂ F₃
      φψ : Quiver.Hom F₁ F₃
      fac : Eq (CategoryTheory.CategoryStruct.comp φ ψ) φψ
      inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J ψ
      ⊢ CategoryTheory.Presheaf.IsLocallyInjective J φ → CategoryTheory.Presheaf.IsL …
    -/
  · intro
    /-
      case mpr
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      inst✝¹ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      ψ : Quiver.Hom F₂ F₃
      φψ : Quiver.Hom F₁ F₃
      fac : Eq (CategoryTheory.CategoryStruct.comp φ ψ) φψ
      inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J ψ
      a✝ : CategoryTheory.Presheaf.IsLocallyInjective J φ
      ⊢ CategoryTheory.Presheaf.IsLocallyInjective J φψ
    -/
    rw [← fac]
    /-
      case mpr
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      inst✝¹ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      ψ : Quiver.Hom F₂ F₃
      φψ : Quiver.Hom F₁ F₃
      fac : Eq (CategoryTheory.CategoryStruct.comp φ ψ) φψ
      inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J ψ
      a✝ : CategoryTheory.Presheaf.IsLocallyInjective J φ
      ⊢ CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.CategoryStruct. …
    -/
    infer_instance
    /-
      🎉 no goals
    -/


lemma isLocallyInjective_comp_iff [IsLocallyInjective J ψ] :
    IsLocallyInjective J (φ ≫ ψ) ↔ IsLocallyInjective J φ :=
  isLocallyInjective_iff_of_fac J rfl


lemma isLocallyInjective_iff_injective_of_separated
    (hsep : Presieve.IsSeparated J (F₁ ⋙ forget D)) :
    IsLocallyInjective J φ ↔ ∀ (X : Cᵒᵖ), Function.Injective (φ.app X) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} D
    inst✝ : CategoryTheory.ConcreteCategory D
    J : CategoryTheory.GrothendieckTopology C
    F₁ F₂ : CategoryTheory.Functor (Opposite C) D
    φ : Quiver.Hom F₁ F₂
    hsep : CategoryTheory.Presieve.IsSeparated J (F₁.comp (CategoryTheory.forget D))
    ⊢ Iff (CategoryTheory.Presheaf.IsLocallyInjective J φ) (∀ (X : Opposite C), Fu …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      hsep : CategoryTheory.Presieve.IsSeparated J (F₁.comp (CategoryTheory.forget D))
      ⊢ CategoryTheory.Presheaf.IsLocallyInjective J φ → ∀ (X : Opposite C), Functio …
    -/
  · intro _ X x y h
    /-
      case mp
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      hsep : CategoryTheory.Presieve.IsSeparated J (F₁.comp (CategoryTheory.forget D))
      a✝ : CategoryTheory.Presheaf.IsLocallyInjective J φ
      X : Opposite C
      x y : (CategoryTheory.forget D).obj (F₁.obj X)
      h : Eq ((φ.app X) x) ((φ.app X) y)
      ⊢ Eq x y
    -/
    exact (hsep _ (equalizerSieve_mem J φ x y h)).ext (fun _ _ hf => hf)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      hsep : CategoryTheory.Presieve.IsSeparated J (F₁.comp (CategoryTheory.forget D))
      ⊢ (∀ (X : Opposite C), Function.Injective ⇑(φ.app X)) → CategoryTheory.Preshea …
    -/
  · apply isLocallyInjective_of_injective
    /-
      🎉 no goals
    -/


instance (F : Cᵒᵖ ⥤ Type w) (G : GrothendieckTopology.Subpresheaf F) :
    IsLocallyInjective J G.ι :=
  isLocallyInjective_of_injective _ _ (fun X => by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      ψ : Quiver.Hom F₂ F₃
      F : CategoryTheory.Functor (Opposite C) (Type w)
      G : CategoryTheory.GrothendieckTopology.Subpresheaf F
      X : Opposite C
      ⊢ Function.Injective ⇑(G.ι.app X)
    -/
    intro ⟨x, _⟩ ⟨y, _⟩ h
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      ψ : Quiver.Hom F₂ F₃
      F : CategoryTheory.Functor (Opposite C) (Type w)
      G : CategoryTheory.GrothendieckTopology.Subpresheaf F
      X : Opposite C
      x : F.obj X
      property✝¹ : Membership.mem (G.obj X) x
      y : F.obj X
      property✝ : Membership.mem (G.obj X) y
      h : Eq ((G.ι.app X) ⟨x, property✝¹⟩) ((G.ι.app X) ⟨y, property✝⟩)
      ⊢ Eq ⟨x, property✝¹⟩ ⟨y, property✝⟩
    -/
    exact Subtype.ext h)
    /-
      🎉 no goals
    -/


instance isLocallyInjective_toPlus (P : Cᵒᵖ ⥤ Type max u v) :
    IsLocallyInjective J (J.toPlus P) where
  equalizerSieve_mem {X} x y h := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      ψ : Quiver.Hom F₂ F₃
      P : CategoryTheory.Functor (Opposite C) (Type (max u v))
      X : Opposite C
      x y : (CategoryTheory.forget (Type (max u v))).obj (P.obj X)
      h : Eq (((J.toPlus P).app X) x) (((J.toPlus P).app X) y)
      ⊢ Membership.mem (J (Opposite.unop X)) (CategoryTheory.Presheaf.equalizerSieve …
    -/
    rw [toPlus_eq_mk, toPlus_eq_mk, eq_mk_iff_exists] at h
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      ψ : Quiver.Hom F₂ F₃
      P : CategoryTheory.Functor (Opposite C) (Type (max u v))
      X : Opposite C
      x y : (CategoryTheory.forget (Type (max u v))).obj (P.obj X)
      h : Exists fun W => Exists fun h1 => Exists fun h2 => Eq ((CategoryTheory.Meq. …
      ⊢ Membership.mem (J (Opposite.unop X)) (CategoryTheory.Presheaf.equalizerSieve …
    -/
    obtain ⟨W, h₁, h₂, eq⟩ := h
    /-
      case intro.intro.intro
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) D
      φ : Quiver.Hom F₁ F₂
      ψ : Quiver.Hom F₂ F₃
      P : CategoryTheory.Functor (Opposite C) (Type (max u v))
      X : Opposite C
      x y : (CategoryTheory.forget (Type (max u v))).obj (P.obj X)
      W : J.Cover (Opposite.unop X)
      h₁ h₂ : Quiver.Hom W Top.top
      eq : Eq ((CategoryTheory.Meq.mk Top.top x).refine h₁) ((CategoryTheory.Meq.mk  …
      ⊢ Membership.mem (J (Opposite.unop X)) (CategoryTheory.Presheaf.equalizerSieve …
    -/
    exact J.superset_covering (fun Y f hf => congr_fun (congr_arg Subtype.val eq) ⟨Y, f, hf⟩) W.2
    /-
      🎉 no goals
    -/


instance isLocallyInjective_toSheafify (P : Cᵒᵖ ⥤ Type max u v) :
    IsLocallyInjective J (J.toSheafify P) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} D
    inst✝ : CategoryTheory.ConcreteCategory D
    J : CategoryTheory.GrothendieckTopology C
    F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) D
    φ : Quiver.Hom F₁ F₂
    ψ : Quiver.Hom F₂ F₃
    P : CategoryTheory.Functor (Opposite C) (Type (max u v))
    ⊢ CategoryTheory.Presheaf.IsLocallyInjective J (J.toSheafify P)
  -/
  dsimp [GrothendieckTopology.toSheafify]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} D
    inst✝ : CategoryTheory.ConcreteCategory D
    J : CategoryTheory.GrothendieckTopology C
    F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) D
    φ : Quiver.Hom F₁ F₂
    ψ : Quiver.Hom F₂ F₃
    P : CategoryTheory.Functor (Opposite C) (Type (max u v))
    ⊢ CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.CategoryStruct. …
  -/
  rw [GrothendieckTopology.plusMap_toPlus]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} D
    inst✝ : CategoryTheory.ConcreteCategory D
    J : CategoryTheory.GrothendieckTopology C
    F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) D
    φ : Quiver.Hom F₁ F₂
    ψ : Quiver.Hom F₂ F₃
    P : CategoryTheory.Functor (Opposite C) (Type (max u v))
    ⊢ CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.CategoryStruct. …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance isLocallyInjective_toSheafify' [ConcreteCategory.{max u v} D]
    (P : Cᵒᵖ ⥤ D) [HasWeakSheafify J D] [J.HasSheafCompose (forget D)]
    [J.PreservesSheafification (forget D)] :
    IsLocallyInjective J (toSheafify J P) := by
  rw [← isLocallyInjective_forget_iff, ← sheafComposeIso_hom_fac,
    ← toSheafify_plusPlusIsoSheafify_hom]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v', u'} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    J : CategoryTheory.GrothendieckTopology C
    F₁ F₂ F₃ : CategoryTheory.Functor (Opposite C) D
    φ : Quiver.Hom F₁ F₂
    ψ : Quiver.Hom F₂ F₃
    inst✝³ : CategoryTheory.ConcreteCategory D
    P : CategoryTheory.Functor (Opposite C) D
    inst✝² : CategoryTheory.HasWeakSheafify J D
    inst✝¹ : J.HasSheafCompose (CategoryTheory.forget D)
    inst✝ : J.PreservesSheafification (CategoryTheory.forget D)
    ⊢ CategoryTheory.Presheaf.IsLocallyInjective J (CategoryTheory.CategoryStruct. …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- If `φ : F₁ ⟶ F₂` is a morphism of sheaves, this is an abbreviation for
`Presheaf.IsLocallyInjective J φ.val`. Under suitable assumptions, it
is equivalent to the injectivity of all maps `φ.val.app X`,
see `isLocallyInjective_iff_injective`. -/
abbrev IsLocallyInjective := Presheaf.IsLocallyInjective J φ.val


lemma isLocallyInjective_sheafToPresheaf_map_iff :
                                                                                             /-
                                                                                               C : Type u
                                                                                               inst✝² : CategoryTheory.Category.{v, u} C
                                                                                               D : Type u'
                                                                                               inst✝¹ : CategoryTheory.Category.{v', u'} D
                                                                                               inst✝ : CategoryTheory.ConcreteCategory D
                                                                                               J : CategoryTheory.GrothendieckTopology C
                                                                                               F₁ F₂ : CategoryTheory.Sheaf J D
                                                                                               φ : Quiver.Hom F₁ F₂
                                                                                               ⊢ Iff (CategoryTheory.Presheaf.IsLocallyInjective J ((CategoryTheory.sheafToPr …
                                                                                             -/
    Presheaf.IsLocallyInjective J ((sheafToPresheaf J D).map φ) ↔ IsLocallyInjective φ := by rfl
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/


instance isLocallyInjective_of_iso [IsIso φ] : IsLocallyInjective φ := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} D
    inst✝¹ : CategoryTheory.ConcreteCategory D
    J : CategoryTheory.GrothendieckTopology C
    F₁ F₂ : CategoryTheory.Sheaf J D
    φ : Quiver.Hom F₁ F₂
    inst✝ : CategoryTheory.IsIso φ
    ⊢ CategoryTheory.Sheaf.IsLocallyInjective φ
  -/
  change Presheaf.IsLocallyInjective J ((sheafToPresheaf _ _).map φ)
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} D
    inst✝¹ : CategoryTheory.ConcreteCategory D
    J : CategoryTheory.GrothendieckTopology C
    F₁ F₂ : CategoryTheory.Sheaf J D
    φ : Quiver.Hom F₁ F₂
    inst✝ : CategoryTheory.IsIso φ
    ⊢ CategoryTheory.Presheaf.IsLocallyInjective J ((CategoryTheory.sheafToPreshea …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma mono_of_injective
    (hφ : ∀ (X : Cᵒᵖ), Function.Injective (φ.val.app X)) : Mono φ :=
  have := fun X ↦ ConcreteCategory.mono_of_injective _ (hφ X)
  (sheafToPresheaf _ _).mono_of_mono_map (NatTrans.mono_of_mono_app φ.1)


instance isLocallyInjective_forget [IsLocallyInjective φ] :
    IsLocallyInjective ((sheafCompose J (forget D)).map φ) :=
  Presheaf.isLocallyInjective_forget J φ.1


lemma isLocallyInjective_iff_injective :
    IsLocallyInjective φ ↔ ∀ (X : Cᵒᵖ), Function.Injective (φ.val.app X) :=
  Presheaf.isLocallyInjective_iff_injective_of_separated _ _ (by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      inst✝¹ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ : CategoryTheory.Sheaf J D
      φ : Quiver.Hom F₁ F₂
      inst✝ : J.HasSheafCompose (CategoryTheory.forget D)
      ⊢ CategoryTheory.Presieve.IsSeparated J (F₁.val.comp (CategoryTheory.forget D))
    -/
    apply Presieve.isSeparated_of_isSheaf
    /-
      case h
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      inst✝¹ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ : CategoryTheory.Sheaf J D
      φ : Quiver.Hom F₁ F₂
      inst✝ : J.HasSheafCompose (CategoryTheory.forget D)
      ⊢ CategoryTheory.Presieve.IsSheaf J (F₁.val.comp (CategoryTheory.forget D))
    -/
    rw [← isSheaf_iff_isSheaf_of_type]
    /-
      case h
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      inst✝¹ : CategoryTheory.ConcreteCategory D
      J : CategoryTheory.GrothendieckTopology C
      F₁ F₂ : CategoryTheory.Sheaf J D
      φ : Quiver.Hom F₁ F₂
      inst✝ : J.HasSheafCompose (CategoryTheory.forget D)
      ⊢ CategoryTheory.Presheaf.IsSheaf J (F₁.val.comp (CategoryTheory.forget D))
    -/
    exact ((sheafCompose J (forget D)).obj F₁).2)
    /-
      🎉 no goals
    -/


lemma mono_of_isLocallyInjective [IsLocallyInjective φ] : Mono φ := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    inst✝² : CategoryTheory.ConcreteCategory D
    J : CategoryTheory.GrothendieckTopology C
    F₁ F₂ : CategoryTheory.Sheaf J D
    φ : Quiver.Hom F₁ F₂
    inst✝¹ : J.HasSheafCompose (CategoryTheory.forget D)
    inst✝ : CategoryTheory.Sheaf.IsLocallyInjective φ
    ⊢ CategoryTheory.Mono φ
  -/
  apply mono_of_injective
  /-
    case hφ
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    inst✝² : CategoryTheory.ConcreteCategory D
    J : CategoryTheory.GrothendieckTopology C
    F₁ F₂ : CategoryTheory.Sheaf J D
    φ : Quiver.Hom F₁ F₂
    inst✝¹ : J.HasSheafCompose (CategoryTheory.forget D)
    inst✝ : CategoryTheory.Sheaf.IsLocallyInjective φ
    ⊢ ∀ (X : Opposite C), Function.Injective ⇑(φ.val.app X)
  -/
  rw [← isLocallyInjective_iff_injective]
  /-
    case hφ
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    inst✝² : CategoryTheory.ConcreteCategory D
    J : CategoryTheory.GrothendieckTopology C
    F₁ F₂ : CategoryTheory.Sheaf J D
    φ : Quiver.Hom F₁ F₂
    inst✝¹ : J.HasSheafCompose (CategoryTheory.forget D)
    inst✝ : CategoryTheory.Sheaf.IsLocallyInjective φ
    ⊢ CategoryTheory.Sheaf.IsLocallyInjective φ
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance {F G : Sheaf J (Type w)} (f : F ⟶ G) :
    IsLocallyInjective (GrothendieckTopology.imageSheafι f) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} D
    inst✝¹ : CategoryTheory.ConcreteCategory D
    J : CategoryTheory.GrothendieckTopology C
    F₁ F₂ : CategoryTheory.Sheaf J D
    φ : Quiver.Hom F₁ F₂
    inst✝ : J.HasSheafCompose (CategoryTheory.forget D)
    F G : CategoryTheory.Sheaf J (Type w)
    f : Quiver.Hom F G
    ⊢ CategoryTheory.Sheaf.IsLocallyInjective (CategoryTheory.GrothendieckTopology …
  -/
  dsimp [GrothendieckTopology.imageSheafι]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} D
    inst✝¹ : CategoryTheory.ConcreteCategory D
    J : CategoryTheory.GrothendieckTopology C
    F₁ F₂ : CategoryTheory.Sheaf J D
    φ : Quiver.Hom F₁ F₂
    inst✝ : J.HasSheafCompose (CategoryTheory.forget D)
    F G : CategoryTheory.Sheaf J (Type w)
    f : Quiver.Hom F G
    ⊢ CategoryTheory.Sheaf.IsLocallyInjective { val := (CategoryTheory.Grothendiec …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


