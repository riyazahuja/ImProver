/-- Given a predicate `P : C → Prop`, this is the class of morphisms `f : X ⟶ Y`
such that for all `Z : C` such that `P Z`, the precomposition with `f` induces
a bijection `(Y ⟶ Z) ≃ (X ⟶ Z)`. -/
def W : MorphismProperty C := fun _ _ f =>
  ∀ Z, P Z → Function.Bijective (fun (g : _ ⟶ Z) => f ≫ g)


variable {P} in
/-- The bijection `(Y ⟶ Z) ≃ (X ⟶ Z)` induced by `f : X ⟶ Y` when `LeftBousfield.W P f`
and `P Z`. -/
@[simps! apply]
noncomputable def W.homEquiv {X Y : C} {f : X ⟶ Y} (hf : W P f) (Z : C) (hZ : P Z) :
    (Y ⟶ Z) ≃ (X ⟶ Z) :=
  Equiv.ofBijective _ (hf Z hZ)


lemma W_isoClosure : W (isoClosure P) = W P := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_3, u_1} C
    P : C → Prop
    ⊢ Eq (CategoryTheory.Localization.LeftBousfield.W (CategoryTheory.isoClosure P …
  -/
  ext X Y f
  /-
    case h
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_3, u_1} C
    P : C → Prop
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.Localization.LeftBousfield.W (CategoryTheory.isoClosure  …
  -/
  constructor
    /-
      case h.mp
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_3, u_1} C
      P : C → Prop
      X Y : C
      f : Quiver.Hom X Y
      ⊢ CategoryTheory.Localization.LeftBousfield.W (CategoryTheory.isoClosure P) f  …
    -/
  · intro hf Z hZ
    /-
      case h.mp
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_3, u_1} C
      P : C → Prop
      X Y : C
      f : Quiver.Hom X Y
      hf : CategoryTheory.Localization.LeftBousfield.W (CategoryTheory.isoClosure P) f
      Z : C
      hZ : P Z
      ⊢ Function.Bijective fun g => CategoryTheory.CategoryStruct.comp f g
    -/
    exact hf _ (le_isoClosure _ _ hZ)
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_3, u_1} C
      P : C → Prop
      X Y : C
      f : Quiver.Hom X Y
      ⊢ CategoryTheory.Localization.LeftBousfield.W P f → CategoryTheory.Localizatio …
    -/
  · rintro hf Z ⟨Z', hZ', ⟨e⟩⟩
    /-
      case h.mpr.intro.intro.intro
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_3, u_1} C
      P : C → Prop
      X Y : C
      f : Quiver.Hom X Y
      hf : CategoryTheory.Localization.LeftBousfield.W P f
      Z Z' : C
      hZ' : P Z'
      e : CategoryTheory.Iso Z Z'
      ⊢ Function.Bijective fun g => CategoryTheory.CategoryStruct.comp f g
    -/
    constructor
      /-
        case h.mpr.intro.intro.intro.left
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_3, u_1} C
        P : C → Prop
        X Y : C
        f : Quiver.Hom X Y
        hf : CategoryTheory.Localization.LeftBousfield.W P f
        Z Z' : C
        hZ' : P Z'
        e : CategoryTheory.Iso Z Z'
        ⊢ Function.Injective fun g => CategoryTheory.CategoryStruct.comp f g
      -/
    · intro g₁ g₂ eq
      /-
        case h.mpr.intro.intro.intro.left
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_3, u_1} C
        P : C → Prop
        X Y : C
        f : Quiver.Hom X Y
        hf : CategoryTheory.Localization.LeftBousfield.W P f
        Z Z' : C
        hZ' : P Z'
        e : CategoryTheory.Iso Z Z'
        g₁ g₂ : Quiver.Hom Y Z
        eq : Eq ((fun g => CategoryTheory.CategoryStruct.comp f g) g₁) ((fun g => Cate …
        ⊢ Eq g₁ g₂
      -/
      rw [← cancel_mono e.hom]
      /-
        case h.mpr.intro.intro.intro.left
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_3, u_1} C
        P : C → Prop
        X Y : C
        f : Quiver.Hom X Y
        hf : CategoryTheory.Localization.LeftBousfield.W P f
        Z Z' : C
        hZ' : P Z'
        e : CategoryTheory.Iso Z Z'
        g₁ g₂ : Quiver.Hom Y Z
        eq : Eq ((fun g => CategoryTheory.CategoryStruct.comp f g) g₁) ((fun g => Cate …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp g₁ e.hom) (CategoryTheory.CategoryStr …
      -/
      apply (hf _ hZ').1
      /-
        case h.mpr.intro.intro.intro.left.a
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_3, u_1} C
        P : C → Prop
        X Y : C
        f : Quiver.Hom X Y
        hf : CategoryTheory.Localization.LeftBousfield.W P f
        Z Z' : C
        hZ' : P Z'
        e : CategoryTheory.Iso Z Z'
        g₁ g₂ : Quiver.Hom Y Z
        eq : Eq ((fun g => CategoryTheory.CategoryStruct.comp f g) g₁) ((fun g => Cate …
        ⊢ Eq ((fun g => CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.Catego …
      -/
      simp only [reassoc_of% eq]
      /-
        🎉 no goals
      -/
      /-
        case h.mpr.intro.intro.intro.right
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_3, u_1} C
        P : C → Prop
        X Y : C
        f : Quiver.Hom X Y
        hf : CategoryTheory.Localization.LeftBousfield.W P f
        Z Z' : C
        hZ' : P Z'
        e : CategoryTheory.Iso Z Z'
        ⊢ Function.Surjective fun g => CategoryTheory.CategoryStruct.comp f g
      -/
    · intro g
      /-
        case h.mpr.intro.intro.intro.right
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_3, u_1} C
        P : C → Prop
        X Y : C
        f : Quiver.Hom X Y
        hf : CategoryTheory.Localization.LeftBousfield.W P f
        Z Z' : C
        hZ' : P Z'
        e : CategoryTheory.Iso Z Z'
        g : Quiver.Hom X Z
        ⊢ Exists fun a => Eq ((fun g => CategoryTheory.CategoryStruct.comp f g) a) g
      -/
      obtain ⟨a, h⟩ := (hf _ hZ').2 (g ≫ e.hom)
      /-
        case h.mpr.intro.intro.intro.right.intro
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_3, u_1} C
        P : C → Prop
        X Y : C
        f : Quiver.Hom X Y
        hf : CategoryTheory.Localization.LeftBousfield.W P f
        Z Z' : C
        hZ' : P Z'
        e : CategoryTheory.Iso Z Z'
        g : Quiver.Hom X Z
        a : Quiver.Hom Y Z'
        h : Eq ((fun g => CategoryTheory.CategoryStruct.comp f g) a) (CategoryTheory.C …
        ⊢ Exists fun a => Eq ((fun g => CategoryTheory.CategoryStruct.comp f g) a) g
      -/
      exact ⟨a ≫ e.inv, by simp only [reassoc_of% h, e.hom_inv_id, comp_id]⟩
      /-
        🎉 no goals
      -/


instance : (W P).IsMultiplicative where
                     /-
                       C : Type u_1
                       D : Type u_2
                       inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
                       inst✝ : CategoryTheory.Category.{?u.2172, u_2} D
                       P : C → Prop
                       X Z : C
                       x✝ : P Z
                       ⊢ Function.Bijective fun g => CategoryTheory.CategoryStruct.comp (CategoryTheo …
                     -/
  id_mem X Z _ := by simpa [id_comp] using Function.bijective_id
                     /-
                       🎉 no goals
                     -/
  comp_mem f g hf hg Z hZ := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Category.{?u.2172, u_2} D
      P : C → Prop
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      hf : CategoryTheory.Localization.LeftBousfield.W P f
      hg : CategoryTheory.Localization.LeftBousfield.W P g
      Z : C
      hZ : P Z
      ⊢ Function.Bijective fun g_1 => CategoryTheory.CategoryStruct.comp (CategoryTh …
    -/
    simpa using Function.Bijective.comp (hf Z hZ) (hg Z hZ)
    /-
      🎉 no goals
    -/


instance : (W P).HasTwoOutOfThreeProperty where
  of_postcomp f g hg hfg Z hZ := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Category.{?u.2630, u_2} D
      P : C → Prop
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      hg : CategoryTheory.Localization.LeftBousfield.W P g
      hfg : CategoryTheory.Localization.LeftBousfield.W P (CategoryTheory.CategorySt …
      Z : C
      hZ : P Z
      ⊢ Function.Bijective fun g => CategoryTheory.CategoryStruct.comp f g
    -/
    rw [← Function.Bijective.of_comp_iff _ (hg Z hZ)]
    /-
      C : Type u_1
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Category.{?u.2630, u_2} D
      P : C → Prop
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      hg : CategoryTheory.Localization.LeftBousfield.W P g
      hfg : CategoryTheory.Localization.LeftBousfield.W P (CategoryTheory.CategorySt …
      Z : C
      hZ : P Z
      ⊢ Function.Bijective (Function.comp (fun g => CategoryTheory.CategoryStruct.co …
    -/
    simpa using hfg Z hZ
    /-
      🎉 no goals
    -/
  of_precomp f g hf hfg Z hZ := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Category.{?u.2630, u_2} D
      P : C → Prop
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      hf : CategoryTheory.Localization.LeftBousfield.W P f
      hfg : CategoryTheory.Localization.LeftBousfield.W P (CategoryTheory.CategorySt …
      Z : C
      hZ : P Z
      ⊢ Function.Bijective fun g_1 => CategoryTheory.CategoryStruct.comp g g_1
    -/
    rw [← Function.Bijective.of_comp_iff' (hf Z hZ)]
    /-
      C : Type u_1
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Category.{?u.2630, u_2} D
      P : C → Prop
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      hf : CategoryTheory.Localization.LeftBousfield.W P f
      hfg : CategoryTheory.Localization.LeftBousfield.W P (CategoryTheory.CategorySt …
      Z : C
      hZ : P Z
      ⊢ Function.Bijective (Function.comp (fun g => CategoryTheory.CategoryStruct.co …
    -/
    simpa using hfg Z hZ
    /-
      🎉 no goals
    -/


lemma W_of_isIso {X Y : C} (f : X ⟶ Y) [IsIso f] : W P f := fun Z _ => by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    P : C → Prop
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f
    Z : C
    x✝ : P Z
    ⊢ Function.Bijective fun g => CategoryTheory.CategoryStruct.comp f g
  -/
  constructor
    /-
      case left
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      P : C → Prop
      X Y : C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.IsIso f
      Z : C
      x✝ : P Z
      ⊢ Function.Injective fun g => CategoryTheory.CategoryStruct.comp f g
    -/
  · intro g₁ g₂ _
    /-
      case left
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      P : C → Prop
      X Y : C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.IsIso f
      Z : C
      x✝ : P Z
      g₁ g₂ : Quiver.Hom Y Z
      a✝ : Eq ((fun g => CategoryTheory.CategoryStruct.comp f g) g₁) ((fun g => Cate …
      ⊢ Eq g₁ g₂
    -/
    simpa only [← cancel_epi f]
    /-
      🎉 no goals
    -/
    /-
      case right
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      P : C → Prop
      X Y : C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.IsIso f
      Z : C
      x✝ : P Z
      ⊢ Function.Surjective fun g => CategoryTheory.CategoryStruct.comp f g
    -/
  · intro g
    /-
      case right
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      P : C → Prop
      X Y : C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.IsIso f
      Z : C
      x✝ : P Z
      g : Quiver.Hom X Z
      ⊢ Exists fun a => Eq ((fun g => CategoryTheory.CategoryStruct.comp f g) a) g
    -/
    exact ⟨inv f ≫ g, by simp⟩
    /-
      🎉 no goals
    -/


lemma W_iff_isIso {X Y : C} (f : X ⟶ Y) (hX : P X) (hY : P Y) :
    W P f ↔ IsIso f := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_3, u_1} C
    P : C → Prop
    X Y : C
    f : Quiver.Hom X Y
    hX : P X
    hY : P Y
    ⊢ Iff (CategoryTheory.Localization.LeftBousfield.W P f) (CategoryTheory.IsIso f)
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_3, u_1} C
      P : C → Prop
      X Y : C
      f : Quiver.Hom X Y
      hX : P X
      hY : P Y
      ⊢ CategoryTheory.Localization.LeftBousfield.W P f → CategoryTheory.IsIso f
    -/
  · intro hf
    /-
      case mp
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_3, u_1} C
      P : C → Prop
      X Y : C
      f : Quiver.Hom X Y
      hX : P X
      hY : P Y
      hf : CategoryTheory.Localization.LeftBousfield.W P f
      ⊢ CategoryTheory.IsIso f
    -/
    obtain ⟨g, hg⟩ := (hf _ hX).2 (𝟙 X)
    /-
      case mp.intro
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_3, u_1} C
      P : C → Prop
      X Y : C
      f : Quiver.Hom X Y
      hX : P X
      hY : P Y
      hf : CategoryTheory.Localization.LeftBousfield.W P f
      g : Quiver.Hom Y X
      hg : Eq ((fun g => CategoryTheory.CategoryStruct.comp f g) g) (CategoryTheory. …
      ⊢ CategoryTheory.IsIso f
    -/
    exact ⟨g, hg, (hf _ hY).1 (by simp only [reassoc_of% hg, comp_id])⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_3, u_1} C
      P : C → Prop
      X Y : C
      f : Quiver.Hom X Y
      hX : P X
      hY : P Y
      ⊢ CategoryTheory.IsIso f → CategoryTheory.Localization.LeftBousfield.W P f
    -/
  · apply W_of_isIso
    /-
      🎉 no goals
    -/


lemma W_adj_unit_app (X : D) : W (· ∈ Set.range F.obj) (adj.unit.app X) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction G F
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    X : D
    ⊢ CategoryTheory.Localization.LeftBousfield.W (fun x => Membership.mem (Set.ra …
  -/
  rintro _ ⟨Y, rfl⟩
  convert ((Functor.FullyFaithful.ofFullyFaithful F).homEquiv.symm.trans
    (adj.homEquiv X Y)).bijective using 1
  /-
    case h.e'_3.h
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction G F
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    X : D
    Y : C
    e_1✝ : Eq (Quiver.Hom ((G.comp F).obj X) (F.obj Y)) (Quiver.Hom (F.obj (G.obj  …
    e_2✝ : Eq (Quiver.Hom ((CategoryTheory.Functor.id D).obj X) (F.obj Y)) (Quiver …
    ⊢ Eq (fun g => CategoryTheory.CategoryStruct.comp (adj.unit.app X) g) ⇑((Categ …
  -/
  dsimp [Adjunction.homEquiv]
  /-
    case h.e'_3.h
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction G F
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    X : D
    Y : C
    e_1✝ : Eq (Quiver.Hom ((G.comp F).obj X) (F.obj Y)) (Quiver.Hom (F.obj (G.obj  …
    e_2✝ : Eq (Quiver.Hom ((CategoryTheory.Functor.id D).obj X) (F.obj Y)) (Quiver …
    ⊢ Eq (fun g => CategoryTheory.CategoryStruct.comp (adj.unit.app X) g) (Functio …
  -/
  aesop
  /-
    🎉 no goals
  -/


lemma W_iff_isIso_map {X Y : D} (f : X ⟶ Y) :
    W (· ∈ Set.range F.obj) f ↔ IsIso (G.map f) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction G F
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    X Y : D
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.Localization.LeftBousfield.W (fun x => Membership.mem (S …
  -/
  rw [← (W (· ∈ Set.range F.obj)).postcomp_iff _ _ (W_adj_unit_app adj Y)]
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction G F
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    X Y : D
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.Localization.LeftBousfield.W (fun x => Membership.mem (S …
  -/
  erw [adj.unit.naturality f]
  rw [(W (· ∈ Set.range F.obj)).precomp_iff _ _ (W_adj_unit_app adj X),
    W_iff_isIso _ _ ⟨_, rfl⟩ ⟨_, rfl⟩]
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction G F
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    X Y : D
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.IsIso ((G.comp F).map f)) (CategoryTheory.IsIso (G.map f))
  -/
  constructor
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_4, u_1} C
      inst✝² : CategoryTheory.Category.{u_3, u_2} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction G F
      inst✝¹ : F.Full
      inst✝ : F.Faithful
      X Y : D
      f : Quiver.Hom X Y
      ⊢ CategoryTheory.IsIso ((G.comp F).map f) → CategoryTheory.IsIso (G.map f)
    -/
  · intro h
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_4, u_1} C
      inst✝² : CategoryTheory.Category.{u_3, u_2} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction G F
      inst✝¹ : F.Full
      inst✝ : F.Faithful
      X Y : D
      f : Quiver.Hom X Y
      h : CategoryTheory.IsIso ((G.comp F).map f)
      ⊢ CategoryTheory.IsIso (G.map f)
    -/
    dsimp at h
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_4, u_1} C
      inst✝² : CategoryTheory.Category.{u_3, u_2} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction G F
      inst✝¹ : F.Full
      inst✝ : F.Faithful
      X Y : D
      f : Quiver.Hom X Y
      h : CategoryTheory.IsIso (F.map (G.map f))
      ⊢ CategoryTheory.IsIso (G.map f)
    -/
    exact isIso_of_fully_faithful F (G.map f)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_4, u_1} C
      inst✝² : CategoryTheory.Category.{u_3, u_2} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction G F
      inst✝¹ : F.Full
      inst✝ : F.Faithful
      X Y : D
      f : Quiver.Hom X Y
      ⊢ CategoryTheory.IsIso (G.map f) → CategoryTheory.IsIso ((G.comp F).map f)
    -/
  · intro
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_4, u_1} C
      inst✝² : CategoryTheory.Category.{u_3, u_2} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction G F
      inst✝¹ : F.Full
      inst✝ : F.Faithful
      X Y : D
      f : Quiver.Hom X Y
      a✝ : CategoryTheory.IsIso (G.map f)
      ⊢ CategoryTheory.IsIso ((G.comp F).map f)
    -/
    rw [Functor.comp_map]
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_4, u_1} C
      inst✝² : CategoryTheory.Category.{u_3, u_2} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction G F
      inst✝¹ : F.Full
      inst✝ : F.Faithful
      X Y : D
      f : Quiver.Hom X Y
      a✝ : CategoryTheory.IsIso (G.map f)
      ⊢ CategoryTheory.IsIso (F.map (G.map f))
    -/
    infer_instance
    /-
      🎉 no goals
    -/


lemma W_eq_inverseImage_isomorphisms :
    W (· ∈ Set.range F.obj) = (MorphismProperty.isomorphisms _).inverseImage G := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction G F
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    ⊢ Eq (CategoryTheory.Localization.LeftBousfield.W fun x => Membership.mem (Set …
  -/
  ext P₁ P₂ f
  /-
    case h
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction G F
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    P₁ P₂ : D
    f : Quiver.Hom P₁ P₂
    ⊢ Iff (CategoryTheory.Localization.LeftBousfield.W (fun x => Membership.mem (S …
  -/
  rw [W_iff_isIso_map adj]
  /-
    case h
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction G F
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    P₁ P₂ : D
    f : Quiver.Hom P₁ P₂
    ⊢ Iff (CategoryTheory.IsIso (G.map f)) ((CategoryTheory.MorphismProperty.isomo …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma isLocalization : G.IsLocalization (W (· ∈ Set.range F.obj)) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction G F
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    ⊢ G.IsLocalization (CategoryTheory.Localization.LeftBousfield.W fun x => Membe …
  -/
  rw [W_eq_inverseImage_isomorphisms adj]
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction G F
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    ⊢ G.IsLocalization ((CategoryTheory.MorphismProperty.isomorphisms C).inverseIm …
  -/
  exact adj.isLocalization
  /-
    🎉 no goals
  -/


