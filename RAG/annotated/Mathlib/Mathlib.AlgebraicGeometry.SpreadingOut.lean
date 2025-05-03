/-- The germ map at `x` is injective if there exists some affine `U ∋ x`
  such that the map `Γ(X, U) ⟶ X_x` is injective -/
class Scheme.IsGermInjectiveAt (X : Scheme.{u}) (x : X) : Prop where
  cond : ∃ (U : X.Opens) (hx : x ∈ U), IsAffineOpen U ∧ Function.Injective (X.presheaf.germ U x hx)


lemma injective_germ_basicOpen (U : X.Opens) (hU : IsAffineOpen U)
    (x : X) (hx : x ∈ U) (f : Γ(X, U))
    (hf : x ∈ X.basicOpen f)
    (H : Function.Injective (X.presheaf.germ U x hx)) :
    Function.Injective (X.presheaf.germ (X.basicOpen f) x hf) := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    f : ↑(X.presheaf.obj { unop := U })
    hf : Membership.mem (X.basicOpen f) x
    H : Function.Injective ⇑(X.presheaf.germ U x hx).hom
    ⊢ Function.Injective ⇑(X.presheaf.germ (X.basicOpen f) x hf).hom
  -/
  rw [RingHom.injective_iff_ker_eq_bot, RingHom.ker_eq_bot_iff_eq_zero] at H ⊢
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    f : ↑(X.presheaf.obj { unop := U })
    hf : Membership.mem (X.basicOpen f) x
    H : ∀ (x_1 : ↑(X.presheaf.obj { unop := U })), Eq ((X.presheaf.germ U x hx).ho …
    ⊢ ∀ (x_1 : ↑(X.presheaf.obj { unop := X.basicOpen f })), Eq ((X.presheaf.germ  …
  -/
  intros t ht
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    f : ↑(X.presheaf.obj { unop := U })
    hf : Membership.mem (X.basicOpen f) x
    H : ∀ (x_1 : ↑(X.presheaf.obj { unop := U })), Eq ((X.presheaf.germ U x hx).ho …
    t : ↑(X.presheaf.obj { unop := X.basicOpen f })
    ht : Eq ((X.presheaf.germ (X.basicOpen f) x hf).hom t) 0
    ⊢ Eq t 0
  -/
  have := hU.isLocalization_basicOpen f
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    f : ↑(X.presheaf.obj { unop := U })
    hf : Membership.mem (X.basicOpen f) x
    H : ∀ (x_1 : ↑(X.presheaf.obj { unop := U })), Eq ((X.presheaf.germ U x hx).ho …
    t : ↑(X.presheaf.obj { unop := X.basicOpen f })
    ht : Eq ((X.presheaf.germ (X.basicOpen f) x hf).hom t) 0
    this : IsLocalization.Away f ↑(X.presheaf.obj { unop := X.basicOpen f })
    ⊢ Eq t 0
  -/
  obtain ⟨t, s, rfl⟩ := IsLocalization.mk'_surjective (.powers f) t
  rw [← RingHom.mem_ker, IsLocalization.mk'_eq_mul_mk'_one, Ideal.mul_unit_mem_iff_mem,
    RingHom.mem_ker, RingHom.algebraMap_toAlgebra, CommRingCat.germ_res_apply] at ht
  /-
    case intro.intro
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    f : ↑(X.presheaf.obj { unop := U })
    hf : Membership.mem (X.basicOpen f) x
    H : ∀ (x_1 : ↑(X.presheaf.obj { unop := U })), Eq ((X.presheaf.germ U x hx).ho …
    this : IsLocalization.Away f ↑(X.presheaf.obj { unop := X.basicOpen f })
    t : ↑(X.presheaf.obj { unop := U })
    s : Subtype fun x => Membership.mem (Submonoid.powers f) x
    ht : Eq ((X.presheaf.germ U x ⋯).hom t) 0
    ⊢ Eq (IsLocalization.mk' (↑(X.presheaf.obj { unop := X.basicOpen f })) t s) 0
  -/
  swap; · exact @isUnit_of_invertible _ _ _ (@IsLocalization.invertible_mk'_one ..)
          /-
            🎉 no goals
          -/
  /-
    case intro.intro
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    f : ↑(X.presheaf.obj { unop := U })
    hf : Membership.mem (X.basicOpen f) x
    H : ∀ (x_1 : ↑(X.presheaf.obj { unop := U })), Eq ((X.presheaf.germ U x hx).ho …
    this : IsLocalization.Away f ↑(X.presheaf.obj { unop := X.basicOpen f })
    t : ↑(X.presheaf.obj { unop := U })
    s : Subtype fun x => Membership.mem (Submonoid.powers f) x
    ht : Eq ((X.presheaf.germ U x ⋯).hom t) 0
    ⊢ Eq (IsLocalization.mk' (↑(X.presheaf.obj { unop := X.basicOpen f })) t s) 0
  -/
  rw [H _ ht, IsLocalization.mk'_zero]
  /-
    🎉 no goals
  -/


lemma Scheme.exists_germ_injective (X : Scheme.{u}) (x : X) [X.IsGermInjectiveAt x] :
    ∃ (U : X.Opens) (hx : x ∈ U),
      IsAffineOpen U ∧ Function.Injective (X.presheaf.germ U x hx) :=
  Scheme.IsGermInjectiveAt.cond


lemma Scheme.exists_le_and_germ_injective (X : Scheme.{u}) (x : X) [X.IsGermInjectiveAt x]
    (V : X.Opens) (hxV : x ∈ V) :
    ∃ (U : X.Opens) (hx : x ∈ U),
      IsAffineOpen U ∧ U ≤ V ∧ Function.Injective (X.presheaf.germ U x hx) := by
  /-
    X : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    inst✝ : X.IsGermInjectiveAt x
    V : X.Opens
    hxV : Membership.mem V x
    ⊢ Exists fun U => Exists fun hx => And (AlgebraicGeometry.IsAffineOpen U) (And …
  -/
  obtain ⟨U, hx, hU, H⟩ := Scheme.IsGermInjectiveAt.cond (x := x)
  /-
    case intro.intro.intro
    X : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    inst✝ : X.IsGermInjectiveAt x
    V : X.Opens
    hxV : Membership.mem V x
    U : X.Opens
    hx : Membership.mem U x
    hU : AlgebraicGeometry.IsAffineOpen U
    H : Function.Injective ⇑(X.presheaf.germ U x hx).hom
    ⊢ Exists fun U => Exists fun hx => And (AlgebraicGeometry.IsAffineOpen U) (And …
  -/
  obtain ⟨f, hf, hxf⟩ := hU.exists_basicOpen_le ⟨x, hxV⟩ hx
  /-
    case intro.intro.intro.intro.intro
    X : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    inst✝ : X.IsGermInjectiveAt x
    V : X.Opens
    hxV : Membership.mem V x
    U : X.Opens
    hx : Membership.mem U x
    hU : AlgebraicGeometry.IsAffineOpen U
    H : Function.Injective ⇑(X.presheaf.germ U x hx).hom
    f : ↑(X.presheaf.obj { unop := U })
    hf : LE.le (X.basicOpen f) V
    hxf : Membership.mem (X.basicOpen f) ↑⟨x, hxV⟩
    ⊢ Exists fun U => Exists fun hx => And (AlgebraicGeometry.IsAffineOpen U) (And …
  -/
  exact ⟨X.basicOpen f, hxf, hU.basicOpen f, hf, injective_germ_basicOpen U hU x hx f hxf H⟩
  /-
    🎉 no goals
  -/


instance (x : X) [X.IsGermInjectiveAt x] [IsOpenImmersion f] :
    Y.IsGermInjectiveAt (f.base x) := by
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    R A : CommRingCat
    x : ↑↑X.toPresheafedSpace
    inst✝¹ : X.IsGermInjectiveAt x
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    ⊢ Y.IsGermInjectiveAt (f.base x)
  -/
  obtain ⟨U, hxU, hU, H⟩ := X.exists_germ_injective x
  /-
    case intro.intro.intro
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    R A : CommRingCat
    x : ↑↑X.toPresheafedSpace
    inst✝¹ : X.IsGermInjectiveAt x
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    U : X.Opens
    hxU : Membership.mem U x
    hU : AlgebraicGeometry.IsAffineOpen U
    H : Function.Injective ⇑(X.presheaf.germ U x hxU).hom
    ⊢ Y.IsGermInjectiveAt (f.base x)
  -/
  refine ⟨⟨f ''ᵁ U, ⟨x, hxU, rfl⟩, hU.image_of_isOpenImmersion f, ?_⟩⟩
  refine ((MorphismProperty.injective CommRingCat).cancel_right_of_respectsIso _
    (f.stalkMap x)).mp ?_
  refine ((MorphismProperty.injective CommRingCat).cancel_left_of_respectsIso
    (f.appIso U).inv _).mp ?_
  /-
    case intro.intro.intro
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    R A : CommRingCat
    x : ↑↑X.toPresheafedSpace
    inst✝¹ : X.IsGermInjectiveAt x
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    U : X.Opens
    hxU : Membership.mem U x
    hU : AlgebraicGeometry.IsAffineOpen U
    H : Function.Injective ⇑(X.presheaf.germ U x hxU).hom
    ⊢ CategoryTheory.MorphismProperty.injective CommRingCat (CategoryTheory.Catego …
  -/
  simpa
  /-
    🎉 no goals
  -/


variable {f} in
lemma isGermInjectiveAt_iff_of_isOpenImmersion {x : X} [IsOpenImmersion f] :
    Y.IsGermInjectiveAt (f.base x) ↔ X.IsGermInjectiveAt x := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    x : ↑↑X.toPresheafedSpace
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    ⊢ Iff (Y.IsGermInjectiveAt (f.base x)) (X.IsGermInjectiveAt x)
  -/
  refine ⟨fun H ↦ ?_, fun _ ↦ inferInstance⟩
  obtain ⟨U, hxU, hU, hU', H⟩ :=
    Y.exists_le_and_germ_injective (f.base x) (V := f.opensRange) ⟨x, rfl⟩
  /-
    case intro.intro.intro.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    x : ↑↑X.toPresheafedSpace
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    H✝ : Y.IsGermInjectiveAt (f.base x)
    U : Y.Opens
    hxU : Membership.mem U (f.base x)
    hU : AlgebraicGeometry.IsAffineOpen U
    hU' : LE.le U (AlgebraicGeometry.Scheme.Hom.opensRange f)
    H : Function.Injective ⇑(Y.presheaf.germ U (f.base x) hxU).hom
    ⊢ X.IsGermInjectiveAt x
  -/
  obtain ⟨V, hV⟩ := (IsOpenImmersion.affineOpensEquiv f).surjective ⟨⟨U, hU⟩, hU'⟩
  /-
    case intro.intro.intro.intro.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    x : ↑↑X.toPresheafedSpace
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    H✝ : Y.IsGermInjectiveAt (f.base x)
    U : Y.Opens
    hxU : Membership.mem U (f.base x)
    hU : AlgebraicGeometry.IsAffineOpen U
    hU' : LE.le U (AlgebraicGeometry.Scheme.Hom.opensRange f)
    H : Function.Injective ⇑(Y.presheaf.germ U (f.base x) hxU).hom
    V : ↑X.affineOpens
    hV : Eq ((AlgebraicGeometry.IsOpenImmersion.affineOpensEquiv f) V) ⟨⟨U, hU⟩, h …
    ⊢ X.IsGermInjectiveAt x
  -/
  obtain rfl : f ''ᵁ V = U := Subtype.eq_iff.mp (Subtype.eq_iff.mp hV)
  /-
    case intro.intro.intro.intro.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    x : ↑↑X.toPresheafedSpace
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    H✝ : Y.IsGermInjectiveAt (f.base x)
    V : ↑X.affineOpens
    hxU : Membership.mem ((AlgebraicGeometry.Scheme.Hom.opensFunctor f).obj ↑V) (f …
    hU : AlgebraicGeometry.IsAffineOpen ((AlgebraicGeometry.Scheme.Hom.opensFuncto …
    hU' : LE.le ((AlgebraicGeometry.Scheme.Hom.opensFunctor f).obj ↑V) (AlgebraicG …
    H : Function.Injective ⇑(Y.presheaf.germ ((AlgebraicGeometry.Scheme.Hom.opensF …
    hV : Eq ((AlgebraicGeometry.IsOpenImmersion.affineOpensEquiv f) V) ⟨⟨(Algebrai …
    ⊢ X.IsGermInjectiveAt x
  -/
  obtain ⟨y, hy, e : f.base y = f.base x⟩ := hxU
  /-
    case intro.intro.intro.intro.intro.intro.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    x : ↑↑X.toPresheafedSpace
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    H✝ : Y.IsGermInjectiveAt (f.base x)
    V : ↑X.affineOpens
    hU : AlgebraicGeometry.IsAffineOpen ((AlgebraicGeometry.Scheme.Hom.opensFuncto …
    hU' : LE.le ((AlgebraicGeometry.Scheme.Hom.opensFunctor f).obj ↑V) (AlgebraicG …
    hV : Eq ((AlgebraicGeometry.IsOpenImmersion.affineOpensEquiv f) V) ⟨⟨(Algebrai …
    y : ↑↑X.toPresheafedSpace
    hy : Membership.mem (↑↑V) y
    e : Eq (f.base y) (f.base x)
    H : Function.Injective ⇑(Y.presheaf.germ ((AlgebraicGeometry.Scheme.Hom.opensF …
    ⊢ X.IsGermInjectiveAt x
  -/
  obtain rfl := f.isOpenEmbedding.injective e
  /-
    case intro.intro.intro.intro.intro.intro.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    V : ↑X.affineOpens
    hU : AlgebraicGeometry.IsAffineOpen ((AlgebraicGeometry.Scheme.Hom.opensFuncto …
    hU' : LE.le ((AlgebraicGeometry.Scheme.Hom.opensFunctor f).obj ↑V) (AlgebraicG …
    hV : Eq ((AlgebraicGeometry.IsOpenImmersion.affineOpensEquiv f) V) ⟨⟨(Algebrai …
    y : ↑↑X.toPresheafedSpace
    hy : Membership.mem (↑↑V) y
    H✝ : Y.IsGermInjectiveAt (f.base y)
    e : Eq (f.base y) (f.base y)
    H : Function.Injective ⇑(Y.presheaf.germ ((AlgebraicGeometry.Scheme.Hom.opensF …
    ⊢ X.IsGermInjectiveAt y
  -/
  refine ⟨V, hy, V.2, ?_⟩
  replace H := ((MorphismProperty.injective CommRingCat).cancel_right_of_respectsIso _
    (f.stalkMap y)).mpr H
  replace H := ((MorphismProperty.injective CommRingCat).cancel_left_of_respectsIso
    (f.appIso V).inv _).mpr H
  /-
    case intro.intro.intro.intro.intro.intro.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    V : ↑X.affineOpens
    hU : AlgebraicGeometry.IsAffineOpen ((AlgebraicGeometry.Scheme.Hom.opensFuncto …
    hU' : LE.le ((AlgebraicGeometry.Scheme.Hom.opensFunctor f).obj ↑V) (AlgebraicG …
    hV : Eq ((AlgebraicGeometry.IsOpenImmersion.affineOpensEquiv f) V) ⟨⟨(Algebrai …
    y : ↑↑X.toPresheafedSpace
    hy : Membership.mem (↑↑V) y
    H✝ : Y.IsGermInjectiveAt (f.base y)
    e : Eq (f.base y) (f.base y)
    H : CategoryTheory.MorphismProperty.injective CommRingCat (CategoryTheory.Cate …
    ⊢ Function.Injective ⇑(X.presheaf.germ (↑V) y hy).hom
  -/
  simpa using H
  /-
    🎉 no goals
  -/


/--
The class of schemes such that for each `x : X`,
`Γ(X, U) ⟶ X_x` is injective for some affine `U` containing `x`.

This is typically satisfied when `X` is integral or locally noetherian.
-/
abbrev Scheme.IsGermInjective (X : Scheme.{u}) := ∀ x : X, X.IsGermInjectiveAt x


lemma Scheme.IsGermInjective.of_openCover
    {X : Scheme.{u}} (𝒰 : X.OpenCover) [∀ i, (𝒰.obj i).IsGermInjective] : X.IsGermInjective := by
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    inst✝ : ∀ (i : 𝒰.J), (𝒰.obj i).IsGermInjective
    ⊢ X.IsGermInjective
  -/
  intro x
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    inst✝ : ∀ (i : 𝒰.J), (𝒰.obj i).IsGermInjective
    x : ↑↑X.toPresheafedSpace
    ⊢ X.IsGermInjectiveAt x
  -/
  rw [← (𝒰.covers x).choose_spec]
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    inst✝ : ∀ (i : 𝒰.J), (𝒰.obj i).IsGermInjective
    x : ↑↑X.toPresheafedSpace
    ⊢ X.IsGermInjectiveAt ((𝒰.map (𝒰.f x)).base (Exists.choose ⋯))
  -/
  infer_instance
  /-
    🎉 no goals
  -/


protected
lemma Scheme.IsGermInjective.Spec
    (H : ∀ I : Ideal R, I.IsPrime →
      ∃ f : R, f ∉ I ∧ ∀ (x y : R), y * x = 0 → y ∉ I → ∃ n, f ^ n * x = 0) :
    (Spec R).IsGermInjective := by
  /-
    R : CommRingCat
    H : ∀ (I : Ideal ↑R), I.IsPrime → Exists fun f => And (Not (Membership.mem I f …
    ⊢ (AlgebraicGeometry.Spec R).IsGermInjective
  -/
  refine fun p ↦ ⟨?_⟩
  /-
    R : CommRingCat
    H : ∀ (I : Ideal ↑R), I.IsPrime → Exists fun f => And (Not (Membership.mem I f …
    p : ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace
    ⊢ Exists fun U => Exists fun hx => And (AlgebraicGeometry.IsAffineOpen U) (Fun …
  -/
  obtain ⟨f, hf, H⟩ := H p.asIdeal p.2
  /-
    case intro.intro
    R : CommRingCat
    H✝ : ∀ (I : Ideal ↑R), I.IsPrime → Exists fun f => And (Not (Membership.mem I  …
    p : ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace
    f : ↑R
    hf : Not (Membership.mem p.asIdeal f)
    H : ∀ (x y : ↑R), Eq (HMul.hMul y x) 0 → Not (Membership.mem p.asIdeal y) → Ex …
    ⊢ Exists fun U => Exists fun hx => And (AlgebraicGeometry.IsAffineOpen U) (Fun …
  -/
  refine ⟨PrimeSpectrum.basicOpen f, hf, ?_, ?_⟩
    /-
      case intro.intro.refine_1
      R : CommRingCat
      H✝ : ∀ (I : Ideal ↑R), I.IsPrime → Exists fun f => And (Not (Membership.mem I  …
      p : ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace
      f : ↑R
      hf : Not (Membership.mem p.asIdeal f)
      H : ∀ (x y : ↑R), Eq (HMul.hMul y x) 0 → Not (Membership.mem p.asIdeal y) → Ex …
      ⊢ AlgebraicGeometry.IsAffineOpen (PrimeSpectrum.basicOpen f)
    -/
  · rw [← basicOpen_eq_of_affine]
    /-
      case intro.intro.refine_1
      R : CommRingCat
      H✝ : ∀ (I : Ideal ↑R), I.IsPrime → Exists fun f => And (Not (Membership.mem I  …
      p : ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace
      f : ↑R
      hf : Not (Membership.mem p.asIdeal f)
      H : ∀ (x y : ↑R), Eq (HMul.hMul y x) 0 → Not (Membership.mem p.asIdeal y) → Ex …
      ⊢ AlgebraicGeometry.IsAffineOpen ((AlgebraicGeometry.Spec R).basicOpen ((Algeb …
    -/
    exact (isAffineOpen_top (Spec R)).basicOpen _
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.refine_2
    R : CommRingCat
    H✝ : ∀ (I : Ideal ↑R), I.IsPrime → Exists fun f => And (Not (Membership.mem I  …
    p : ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace
    f : ↑R
    hf : Not (Membership.mem p.asIdeal f)
    H : ∀ (x y : ↑R), Eq (HMul.hMul y x) 0 → Not (Membership.mem p.asIdeal y) → Ex …
    ⊢ Function.Injective ⇑((AlgebraicGeometry.Spec R).presheaf.germ (PrimeSpectrum …
  -/
  rw [RingHom.injective_iff_ker_eq_bot, RingHom.ker_eq_bot_iff_eq_zero]
  /-
    case intro.intro.refine_2
    R : CommRingCat
    H✝ : ∀ (I : Ideal ↑R), I.IsPrime → Exists fun f => And (Not (Membership.mem I  …
    p : ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace
    f : ↑R
    hf : Not (Membership.mem p.asIdeal f)
    H : ∀ (x y : ↑R), Eq (HMul.hMul y x) 0 → Not (Membership.mem p.asIdeal y) → Ex …
    ⊢ ∀ (x : ↑((AlgebraicGeometry.Spec R).presheaf.obj { unop := PrimeSpectrum.bas …
  -/
  intro x hx
  obtain ⟨x, s, rfl⟩ := IsLocalization.mk'_surjective
    (S := ((Spec.structureSheaf R).val.obj (.op <| PrimeSpectrum.basicOpen f))) (.powers f) x
  rw [← RingHom.mem_ker, IsLocalization.mk'_eq_mul_mk'_one, Ideal.mul_unit_mem_iff_mem,
    RingHom.mem_ker, RingHom.algebraMap_toAlgebra] at hx
  /-
    case intro.intro.refine_2.intro.intro
    R : CommRingCat
    H✝ : ∀ (I : Ideal ↑R), I.IsPrime → Exists fun f => And (Not (Membership.mem I  …
    p : ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace
    f : ↑R
    hf : Not (Membership.mem p.asIdeal f)
    H : ∀ (x y : ↑R), Eq (HMul.hMul y x) 0 → Not (Membership.mem p.asIdeal y) → Ex …
    x : ↑R
    s : Subtype fun x => Membership.mem (Submonoid.powers f) x
    hx : Eq (((AlgebraicGeometry.Spec R).presheaf.germ (PrimeSpectrum.basicOpen f) …
    ⊢ Eq (IsLocalization.mk' (↑((AlgebraicGeometry.Spec.structureSheaf ↑R).val.obj …
  -/
  swap; · exact @isUnit_of_invertible _ _ _ (@IsLocalization.invertible_mk'_one ..)
          /-
            🎉 no goals
          -/
  /-
    case intro.intro.refine_2.intro.intro
    R : CommRingCat
    H✝ : ∀ (I : Ideal ↑R), I.IsPrime → Exists fun f => And (Not (Membership.mem I  …
    p : ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace
    f : ↑R
    hf : Not (Membership.mem p.asIdeal f)
    H : ∀ (x y : ↑R), Eq (HMul.hMul y x) 0 → Not (Membership.mem p.asIdeal y) → Ex …
    x : ↑R
    s : Subtype fun x => Membership.mem (Submonoid.powers f) x
    hx : Eq (((AlgebraicGeometry.Spec R).presheaf.germ (PrimeSpectrum.basicOpen f) …
    ⊢ Eq (IsLocalization.mk' (↑((AlgebraicGeometry.Spec.structureSheaf ↑R).val.obj …
  -/
  erw [StructureSheaf.germ_toOpen] at hx
  obtain ⟨⟨y, hy⟩, hy'⟩ := (IsLocalization.map_eq_zero_iff p.asIdeal.primeCompl
    ((Spec.structureSheaf R).presheaf.stalk p) _).mp hx
  /-
    case intro.intro.refine_2.intro.intro.intro.mk
    R : CommRingCat
    H✝ : ∀ (I : Ideal ↑R), I.IsPrime → Exists fun f => And (Not (Membership.mem I  …
    p : ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace
    f : ↑R
    hf : Not (Membership.mem p.asIdeal f)
    H : ∀ (x y : ↑R), Eq (HMul.hMul y x) 0 → Not (Membership.mem p.asIdeal y) → Ex …
    x : ↑R
    s : Subtype fun x => Membership.mem (Submonoid.powers f) x
    hx : Eq ((AlgebraicGeometry.StructureSheaf.toStalk (↑R) p).hom x) 0
    y : ↑R
    hy : Membership.mem p.asIdeal.primeCompl y
    hy' : Eq (HMul.hMul (↑⟨y, hy⟩) x) 0
    ⊢ Eq (IsLocalization.mk' (↑((AlgebraicGeometry.Spec.structureSheaf ↑R).val.obj …
  -/
  obtain ⟨n, hn⟩ := H x y hy' hy
  /-
    case intro.intro.refine_2.intro.intro.intro.mk.intro
    R : CommRingCat
    H✝ : ∀ (I : Ideal ↑R), I.IsPrime → Exists fun f => And (Not (Membership.mem I  …
    p : ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace
    f : ↑R
    hf : Not (Membership.mem p.asIdeal f)
    H : ∀ (x y : ↑R), Eq (HMul.hMul y x) 0 → Not (Membership.mem p.asIdeal y) → Ex …
    x : ↑R
    s : Subtype fun x => Membership.mem (Submonoid.powers f) x
    hx : Eq ((AlgebraicGeometry.StructureSheaf.toStalk (↑R) p).hom x) 0
    y : ↑R
    hy : Membership.mem p.asIdeal.primeCompl y
    hy' : Eq (HMul.hMul (↑⟨y, hy⟩) x) 0
    n : Nat
    hn : Eq (HMul.hMul (HPow.hPow f n) x) 0
    ⊢ Eq (IsLocalization.mk' (↑((AlgebraicGeometry.Spec.structureSheaf ↑R).val.obj …
  -/
  refine (@IsLocalization.mk'_eq_zero_iff ..).mpr ?_
  /-
    case intro.intro.refine_2.intro.intro.intro.mk.intro
    R : CommRingCat
    H✝ : ∀ (I : Ideal ↑R), I.IsPrime → Exists fun f => And (Not (Membership.mem I  …
    p : ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace
    f : ↑R
    hf : Not (Membership.mem p.asIdeal f)
    H : ∀ (x y : ↑R), Eq (HMul.hMul y x) 0 → Not (Membership.mem p.asIdeal y) → Ex …
    x : ↑R
    s : Subtype fun x => Membership.mem (Submonoid.powers f) x
    hx : Eq ((AlgebraicGeometry.StructureSheaf.toStalk (↑R) p).hom x) 0
    y : ↑R
    hy : Membership.mem p.asIdeal.primeCompl y
    hy' : Eq (HMul.hMul (↑⟨y, hy⟩) x) 0
    n : Nat
    hn : Eq (HMul.hMul (HPow.hPow f n) x) 0
    ⊢ Exists fun m => Eq (HMul.hMul (↑m) x) 0
  -/
  exact ⟨⟨_, n, rfl⟩, hn⟩
  /-
    🎉 no goals
  -/


instance (priority := 100) [IsIntegral X] : X.IsGermInjective := by
  refine fun x ↦ ⟨⟨(X.affineCover.map x).opensRange, X.affineCover.covers x,
    (isAffineOpen_opensRange (X.affineCover.map x)), ?_⟩⟩
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    R A : CommRingCat
    inst✝ : AlgebraicGeometry.IsIntegral X
    x : ↑↑X.toPresheafedSpace
    ⊢ Function.Injective ⇑(X.presheaf.germ (AlgebraicGeometry.Scheme.Hom.opensRang …
  -/
  have : Nonempty (X.affineCover.map x).opensRange := ⟨⟨_, X.affineCover.covers x⟩⟩
  have := (isAffineOpen_opensRange (X.affineCover.map x)).isLocalization_stalk
    ⟨_, X.affineCover.covers x⟩
  exact @IsLocalization.injective _ _ _ _ _ (show _ from _) this
    (Ideal.primeCompl_le_nonZeroDivisors _)


instance (priority := 100) [IsLocallyNoetherian X] : X.IsGermInjective := by
  suffices ∀ (R : CommRingCat.{u}) (_ : IsNoetherianRing R), (Spec R).IsGermInjective by
    refine @Scheme.IsGermInjective.of_openCover _ (X.affineOpenCover.openCover) (fun i ↦ this _ ?_)
    have := isLocallyNoetherian_of_isOpenImmersion (X.affineOpenCover.map i)
    infer_instance
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    R A : CommRingCat
    inst✝ : AlgebraicGeometry.IsLocallyNoetherian X
    ⊢ ∀ (R : CommRingCat), IsNoetherianRing ↑R → (AlgebraicGeometry.Spec R).IsGerm …
  -/
  refine fun R hR ↦ Scheme.IsGermInjective.Spec fun I hI ↦ ?_
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    R✝ A : CommRingCat
    inst✝ : AlgebraicGeometry.IsLocallyNoetherian X
    R : CommRingCat
    hR : IsNoetherianRing ↑R
    I : Ideal ↑R
    hI : I.IsPrime
    ⊢ Exists fun f => And (Not (Membership.mem I f)) (∀ (x y : ↑R), Eq (HMul.hMul  …
  -/
  let J := RingHom.ker <| algebraMap R (Localization.AtPrime I)
  have hJ (x) : x ∈ J ↔ ∃ y : I.primeCompl, y * x = 0 :=
    IsLocalization.map_eq_zero_iff I.primeCompl _ x
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    R✝ A : CommRingCat
    inst✝ : AlgebraicGeometry.IsLocallyNoetherian X
    R : CommRingCat
    hR : IsNoetherianRing ↑R
    I : Ideal ↑R
    hI : I.IsPrime
    J : Ideal ↑R := RingHom.ker (algebraMap (↑R) (Localization.AtPrime I))
    hJ : ∀ (x : ↑R), Iff (Membership.mem J x) (Exists fun y => Eq (HMul.hMul (↑y)  …
    ⊢ Exists fun f => And (Not (Membership.mem I f)) (∀ (x y : ↑R), Eq (HMul.hMul  …
  -/
  choose f hf using fun x ↦ (hJ x).mp
  /-
    X Y S : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X Y
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    R✝ A : CommRingCat
    inst✝ : AlgebraicGeometry.IsLocallyNoetherian X
    R : CommRingCat
    hR : IsNoetherianRing ↑R
    I : Ideal ↑R
    hI : I.IsPrime
    J : Ideal ↑R := RingHom.ker (algebraMap (↑R) (Localization.AtPrime I))
    hJ : ∀ (x : ↑R), Iff (Membership.mem J x) (Exists fun y => Eq (HMul.hMul (↑y)  …
    f : (x : ↑R) → Membership.mem J x → Subtype fun x => Membership.mem I.primeCom …
    hf : ∀ (x : ↑R) (a : Membership.mem J x), Eq (HMul.hMul (↑(f x a)) x) 0
    ⊢ Exists fun f => And (Not (Membership.mem I f)) (∀ (x y : ↑R), Eq (HMul.hMul  …
  -/
  obtain ⟨s, hs⟩ := (isNoetherianRing_iff_ideal_fg R).mp ‹_› J
  /-
    case intro
    X Y S : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X Y
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    R✝ A : CommRingCat
    inst✝ : AlgebraicGeometry.IsLocallyNoetherian X
    R : CommRingCat
    hR : IsNoetherianRing ↑R
    I : Ideal ↑R
    hI : I.IsPrime
    J : Ideal ↑R := RingHom.ker (algebraMap (↑R) (Localization.AtPrime I))
    hJ : ∀ (x : ↑R), Iff (Membership.mem J x) (Exists fun y => Eq (HMul.hMul (↑y)  …
    f : (x : ↑R) → Membership.mem J x → Subtype fun x => Membership.mem I.primeCom …
    hf : ∀ (x : ↑R) (a : Membership.mem J x), Eq (HMul.hMul (↑(f x a)) x) 0
    s : Finset ↑R
    hs : Eq (Ideal.span ↑s) J
    ⊢ Exists fun f => And (Not (Membership.mem I f)) (∀ (x y : ↑R), Eq (HMul.hMul  …
  -/
  have hs' : (s : Set R) ⊆ J := hs ▸ Ideal.subset_span
  /-
    case intro
    X Y S : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X Y
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    R✝ A : CommRingCat
    inst✝ : AlgebraicGeometry.IsLocallyNoetherian X
    R : CommRingCat
    hR : IsNoetherianRing ↑R
    I : Ideal ↑R
    hI : I.IsPrime
    J : Ideal ↑R := RingHom.ker (algebraMap (↑R) (Localization.AtPrime I))
    hJ : ∀ (x : ↑R), Iff (Membership.mem J x) (Exists fun y => Eq (HMul.hMul (↑y)  …
    f : (x : ↑R) → Membership.mem J x → Subtype fun x => Membership.mem I.primeCom …
    hf : ∀ (x : ↑R) (a : Membership.mem J x), Eq (HMul.hMul (↑(f x a)) x) 0
    s : Finset ↑R
    hs : Eq (Ideal.span ↑s) J
    hs' : HasSubset.Subset ↑s ↑J
    ⊢ Exists fun f => And (Not (Membership.mem I f)) (∀ (x y : ↑R), Eq (HMul.hMul  …
  -/
  refine ⟨_, (s.attach.prod fun x ↦ f x (hs' x.2)).2, fun x y e hy ↦ ⟨1, ?_⟩⟩
  /-
    case intro
    X Y S : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X Y
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    R✝ A : CommRingCat
    inst✝ : AlgebraicGeometry.IsLocallyNoetherian X
    R : CommRingCat
    hR : IsNoetherianRing ↑R
    I : Ideal ↑R
    hI : I.IsPrime
    J : Ideal ↑R := RingHom.ker (algebraMap (↑R) (Localization.AtPrime I))
    hJ : ∀ (x : ↑R), Iff (Membership.mem J x) (Exists fun y => Eq (HMul.hMul (↑y)  …
    f : (x : ↑R) → Membership.mem J x → Subtype fun x => Membership.mem I.primeCom …
    hf : ∀ (x : ↑R) (a : Membership.mem J x), Eq (HMul.hMul (↑(f x a)) x) 0
    s : Finset ↑R
    hs : Eq (Ideal.span ↑s) J
    hs' : HasSubset.Subset ↑s ↑J
    x y : ↑R
    e : Eq (HMul.hMul y x) 0
    hy : Not (Membership.mem I y)
    ⊢ Eq (HMul.hMul (HPow.hPow (↑(s.attach.prod fun x => f ↑x ⋯)) 1) x) 0
  -/
  rw [pow_one, mul_comm, ← smul_eq_mul, ← Submodule.mem_annihilator_span_singleton]
  /-
    case intro
    X Y S : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X Y
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    R✝ A : CommRingCat
    inst✝ : AlgebraicGeometry.IsLocallyNoetherian X
    R : CommRingCat
    hR : IsNoetherianRing ↑R
    I : Ideal ↑R
    hI : I.IsPrime
    J : Ideal ↑R := RingHom.ker (algebraMap (↑R) (Localization.AtPrime I))
    hJ : ∀ (x : ↑R), Iff (Membership.mem J x) (Exists fun y => Eq (HMul.hMul (↑y)  …
    f : (x : ↑R) → Membership.mem J x → Subtype fun x => Membership.mem I.primeCom …
    hf : ∀ (x : ↑R) (a : Membership.mem J x), Eq (HMul.hMul (↑(f x a)) x) 0
    s : Finset ↑R
    hs : Eq (Ideal.span ↑s) J
    hs' : HasSubset.Subset ↑s ↑J
    x y : ↑R
    e : Eq (HMul.hMul y x) 0
    hy : Not (Membership.mem I y)
    ⊢ Membership.mem (Submodule.span (↑R) (Singleton.singleton ↑(s.attach.prod fun …
  -/
  refine SetLike.le_def.mp ?_ ((hJ x).mpr ⟨⟨y, hy⟩, e⟩)
  /-
    case intro
    X Y S : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X Y
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    R✝ A : CommRingCat
    inst✝ : AlgebraicGeometry.IsLocallyNoetherian X
    R : CommRingCat
    hR : IsNoetherianRing ↑R
    I : Ideal ↑R
    hI : I.IsPrime
    J : Ideal ↑R := RingHom.ker (algebraMap (↑R) (Localization.AtPrime I))
    hJ : ∀ (x : ↑R), Iff (Membership.mem J x) (Exists fun y => Eq (HMul.hMul (↑y)  …
    f : (x : ↑R) → Membership.mem J x → Subtype fun x => Membership.mem I.primeCom …
    hf : ∀ (x : ↑R) (a : Membership.mem J x), Eq (HMul.hMul (↑(f x a)) x) 0
    s : Finset ↑R
    hs : Eq (Ideal.span ↑s) J
    hs' : HasSubset.Subset ↑s ↑J
    x y : ↑R
    e : Eq (HMul.hMul y x) 0
    hy : Not (Membership.mem I y)
    ⊢ LE.le J (Submodule.span (↑R) (Singleton.singleton ↑(s.attach.prod fun x => f …
  -/
  rw [← hs, Ideal.span_le]
  /-
    case intro
    X Y S : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X Y
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    R✝ A : CommRingCat
    inst✝ : AlgebraicGeometry.IsLocallyNoetherian X
    R : CommRingCat
    hR : IsNoetherianRing ↑R
    I : Ideal ↑R
    hI : I.IsPrime
    J : Ideal ↑R := RingHom.ker (algebraMap (↑R) (Localization.AtPrime I))
    hJ : ∀ (x : ↑R), Iff (Membership.mem J x) (Exists fun y => Eq (HMul.hMul (↑y)  …
    f : (x : ↑R) → Membership.mem J x → Subtype fun x => Membership.mem I.primeCom …
    hf : ∀ (x : ↑R) (a : Membership.mem J x), Eq (HMul.hMul (↑(f x a)) x) 0
    s : Finset ↑R
    hs : Eq (Ideal.span ↑s) J
    hs' : HasSubset.Subset ↑s ↑J
    x y : ↑R
    e : Eq (HMul.hMul y x) 0
    hy : Not (Membership.mem I y)
    ⊢ HasSubset.Subset ↑s ↑(Submodule.span (↑R) (Singleton.singleton ↑(s.attach.pr …
  -/
  intro i hi
  rw [SetLike.mem_coe, Submodule.mem_annihilator_span_singleton, smul_eq_mul,
    mul_comm, ← smul_eq_mul, ← Submodule.mem_annihilator_span_singleton, Submonoid.coe_finset_prod]
  /-
    case intro
    X Y S : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X Y
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    R✝ A : CommRingCat
    inst✝ : AlgebraicGeometry.IsLocallyNoetherian X
    R : CommRingCat
    hR : IsNoetherianRing ↑R
    I : Ideal ↑R
    hI : I.IsPrime
    J : Ideal ↑R := RingHom.ker (algebraMap (↑R) (Localization.AtPrime I))
    hJ : ∀ (x : ↑R), Iff (Membership.mem J x) (Exists fun y => Eq (HMul.hMul (↑y)  …
    f : (x : ↑R) → Membership.mem J x → Subtype fun x => Membership.mem I.primeCom …
    hf : ∀ (x : ↑R) (a : Membership.mem J x), Eq (HMul.hMul (↑(f x a)) x) 0
    s : Finset ↑R
    hs : Eq (Ideal.span ↑s) J
    hs' : HasSubset.Subset ↑s ↑J
    x y : ↑R
    e : Eq (HMul.hMul y x) 0
    hy : Not (Membership.mem I y)
    i : ↑R
    hi : Membership.mem (↑s) i
    ⊢ Membership.mem (Submodule.span (↑R) (Singleton.singleton i)).annihilator (s. …
  -/
  refine Ideal.mem_of_dvd _ (Finset.dvd_prod_of_mem _ (s.mem_attach ⟨i, hi⟩)) ?_
  /-
    case intro
    X Y S : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X Y
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    R✝ A : CommRingCat
    inst✝ : AlgebraicGeometry.IsLocallyNoetherian X
    R : CommRingCat
    hR : IsNoetherianRing ↑R
    I : Ideal ↑R
    hI : I.IsPrime
    J : Ideal ↑R := RingHom.ker (algebraMap (↑R) (Localization.AtPrime I))
    hJ : ∀ (x : ↑R), Iff (Membership.mem J x) (Exists fun y => Eq (HMul.hMul (↑y)  …
    f : (x : ↑R) → Membership.mem J x → Subtype fun x => Membership.mem I.primeCom …
    hf : ∀ (x : ↑R) (a : Membership.mem J x), Eq (HMul.hMul (↑(f x a)) x) 0
    s : Finset ↑R
    hs : Eq (Ideal.span ↑s) J
    hs' : HasSubset.Subset ↑s ↑J
    x y : ↑R
    e : Eq (HMul.hMul y x) 0
    hy : Not (Membership.mem I y)
    i : ↑R
    hi : Membership.mem (↑s) i
    ⊢ Membership.mem (Submodule.span (↑R) (Singleton.singleton i)).annihilator ↑(f …
  -/
  rw [Submodule.mem_annihilator_span_singleton, smul_eq_mul]
  /-
    case intro
    X Y S : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X Y
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    R✝ A : CommRingCat
    inst✝ : AlgebraicGeometry.IsLocallyNoetherian X
    R : CommRingCat
    hR : IsNoetherianRing ↑R
    I : Ideal ↑R
    hI : I.IsPrime
    J : Ideal ↑R := RingHom.ker (algebraMap (↑R) (Localization.AtPrime I))
    hJ : ∀ (x : ↑R), Iff (Membership.mem J x) (Exists fun y => Eq (HMul.hMul (↑y)  …
    f : (x : ↑R) → Membership.mem J x → Subtype fun x => Membership.mem I.primeCom …
    hf : ∀ (x : ↑R) (a : Membership.mem J x), Eq (HMul.hMul (↑(f x a)) x) 0
    s : Finset ↑R
    hs : Eq (Ideal.span ↑s) J
    hs' : HasSubset.Subset ↑s ↑J
    x y : ↑R
    e : Eq (HMul.hMul y x) 0
    hy : Not (Membership.mem I y)
    i : ↑R
    hi : Membership.mem (↑s) i
    ⊢ Eq (HMul.hMul (↑(f ↑⟨i, hi⟩ ⋯)) i) 0
  -/
  exact hf i _
  /-
    🎉 no goals
  -/


/--
Let `x : X` and `f g : X ⟶ Y` be two morphisms such that `f x = g x`.
If `f` and `g` agree on the stalk of `x`, then they agree on an open neighborhood of `x`,
provided `X` is "germ-injective" at `x` (e.g. when it's integral or locally noetherian).

TODO: The condition on `X` is unnecessary when `Y` is locally of finite type.
-/
@[stacks 0BX6]
lemma spread_out_unique_of_isGermInjective {x : X} [X.IsGermInjectiveAt x]
    (f g : X ⟶ Y) (e : f.base x = g.base x)
    (H : f.stalkMap x =
      Y.presheaf.stalkSpecializes (Inseparable.of_eq e.symm).specializes ≫ g.stalkMap x) :
    ∃ (U : X.Opens), x ∈ U ∧ U.ι ≫ f = U.ι ≫ g := by
  obtain ⟨_, ⟨V : Y.Opens, hV, rfl⟩, hxV, -⟩ :=
    (isBasis_affine_open Y).exists_subset_of_mem_open (Set.mem_univ (f.base x)) isOpen_univ
  /-
    case intro.intro.intro.intro.intro
    X Y : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    inst✝ : X.IsGermInjectiveAt x
    f g : Quiver.Hom X Y
    e : Eq (f.base x) (g.base x)
    H : Eq (AlgebraicGeometry.Scheme.Hom.stalkMap f x) (CategoryTheory.CategoryStr …
    V : Y.Opens
    hV : Membership.mem Y.affineOpens V
    hxV : Membership.mem (↑V) (f.base x)
    ⊢ Exists fun U => And (Membership.mem U x) (Eq (CategoryTheory.CategoryStruct. …
  -/
  have hxV' : g.base x ∈ V := e ▸ hxV
  /-
    case intro.intro.intro.intro.intro
    X Y : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    inst✝ : X.IsGermInjectiveAt x
    f g : Quiver.Hom X Y
    e : Eq (f.base x) (g.base x)
    H : Eq (AlgebraicGeometry.Scheme.Hom.stalkMap f x) (CategoryTheory.CategoryStr …
    V : Y.Opens
    hV : Membership.mem Y.affineOpens V
    hxV : Membership.mem (↑V) (f.base x)
    hxV' : Membership.mem V (g.base x)
    ⊢ Exists fun U => And (Membership.mem U x) (Eq (CategoryTheory.CategoryStruct. …
  -/
  obtain ⟨U, hxU, _, hUV, HU⟩ := X.exists_le_and_germ_injective x (f ⁻¹ᵁ V ⊓ g ⁻¹ᵁ V) ⟨hxV, hxV'⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    X Y : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    inst✝ : X.IsGermInjectiveAt x
    f g : Quiver.Hom X Y
    e : Eq (f.base x) (g.base x)
    H : Eq (AlgebraicGeometry.Scheme.Hom.stalkMap f x) (CategoryTheory.CategoryStr …
    V : Y.Opens
    hV : Membership.mem Y.affineOpens V
    hxV : Membership.mem (↑V) (f.base x)
    hxV' : Membership.mem V (g.base x)
    U : X.Opens
    hxU : Membership.mem U x
    left✝ : AlgebraicGeometry.IsAffineOpen U
    hUV : LE.le U (Min.min ((TopologicalSpace.Opens.map f.base).obj V) ((Topologic …
    HU : Function.Injective ⇑(X.presheaf.germ U x hxU).hom
    ⊢ Exists fun U => And (Membership.mem U x) (Eq (CategoryTheory.CategoryStruct. …
  -/
  refine ⟨U, hxU, ?_⟩
  rw [← Scheme.Hom.resLE_comp_ι _ (hUV.trans inf_le_left),
    ← Scheme.Hom.resLE_comp_ι _ (hUV.trans inf_le_right)]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    X Y : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    inst✝ : X.IsGermInjectiveAt x
    f g : Quiver.Hom X Y
    e : Eq (f.base x) (g.base x)
    H : Eq (AlgebraicGeometry.Scheme.Hom.stalkMap f x) (CategoryTheory.CategoryStr …
    V : Y.Opens
    hV : Membership.mem Y.affineOpens V
    hxV : Membership.mem (↑V) (f.base x)
    hxV' : Membership.mem V (g.base x)
    U : X.Opens
    hxU : Membership.mem U x
    left✝ : AlgebraicGeometry.IsAffineOpen U
    hUV : LE.le U (Min.min ((TopologicalSpace.Opens.map f.base).obj V) ((Topologic …
    HU : Function.Injective ⇑(X.presheaf.germ U x hxU).hom
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.resLE f …
  -/
  congr 1
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.e_a
    X Y : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    inst✝ : X.IsGermInjectiveAt x
    f g : Quiver.Hom X Y
    e : Eq (f.base x) (g.base x)
    H : Eq (AlgebraicGeometry.Scheme.Hom.stalkMap f x) (CategoryTheory.CategoryStr …
    V : Y.Opens
    hV : Membership.mem Y.affineOpens V
    hxV : Membership.mem (↑V) (f.base x)
    hxV' : Membership.mem V (g.base x)
    U : X.Opens
    hxU : Membership.mem U x
    left✝ : AlgebraicGeometry.IsAffineOpen U
    hUV : LE.le U (Min.min ((TopologicalSpace.Opens.map f.base).obj V) ((Topologic …
    HU : Function.Injective ⇑(X.presheaf.germ U x hxU).hom
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.resLE f V U ⋯) (AlgebraicGeometry.Scheme.Ho …
  -/
  have : IsAffine V := hV
  suffices ∀ (U₀ V₀) (eU : U = U₀) (eV : V = V₀),
      f.appLE V₀ U₀ (eU ▸ eV ▸ hUV.trans inf_le_left) =
        g.appLE V₀ U₀ (eU ▸ eV ▸ hUV.trans inf_le_right) by
    rw [← cancel_mono V.toScheme.isoSpec.hom]
    simp only [Scheme.isoSpec, asIso_hom, Scheme.toSpecΓ_naturality,
      Scheme.Hom.app_eq_appLE, Scheme.Hom.resLE_appLE]
    congr 2
    apply this <;> simp
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.e_a
    X Y : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    inst✝ : X.IsGermInjectiveAt x
    f g : Quiver.Hom X Y
    e : Eq (f.base x) (g.base x)
    H : Eq (AlgebraicGeometry.Scheme.Hom.stalkMap f x) (CategoryTheory.CategoryStr …
    V : Y.Opens
    hV : Membership.mem Y.affineOpens V
    hxV : Membership.mem (↑V) (f.base x)
    hxV' : Membership.mem V (g.base x)
    U : X.Opens
    hxU : Membership.mem U x
    left✝ : AlgebraicGeometry.IsAffineOpen U
    hUV : LE.le U (Min.min ((TopologicalSpace.Opens.map f.base).obj V) ((Topologic …
    HU : Function.Injective ⇑(X.presheaf.germ U x hxU).hom
    this : AlgebraicGeometry.IsAffine ↑V
    ⊢ ∀ (U₀ : X.Opens) (V₀ : Y.Opens) (eU : Eq U U₀) (eV : Eq V V₀), Eq (Algebraic …
  -/
  rintro U V rfl rfl
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.e_a
    X Y : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    inst✝ : X.IsGermInjectiveAt x
    f g : Quiver.Hom X Y
    e : Eq (f.base x) (g.base x)
    H : Eq (AlgebraicGeometry.Scheme.Hom.stalkMap f x) (CategoryTheory.CategoryStr …
    V : Y.Opens
    hV : Membership.mem Y.affineOpens V
    hxV : Membership.mem (↑V) (f.base x)
    hxV' : Membership.mem V (g.base x)
    U : X.Opens
    hxU : Membership.mem U x
    left✝ : AlgebraicGeometry.IsAffineOpen U
    hUV : LE.le U (Min.min ((TopologicalSpace.Opens.map f.base).obj V) ((Topologic …
    HU : Function.Injective ⇑(X.presheaf.germ U x hxU).hom
    this : AlgebraicGeometry.IsAffine ↑V
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.appLE f V U ⋯) (AlgebraicGeometry.Scheme.Ho …
  -/
  have := ConcreteCategory.mono_of_injective (C := CommRingCat) _ HU
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.e_a
    X Y : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    inst✝ : X.IsGermInjectiveAt x
    f g : Quiver.Hom X Y
    e : Eq (f.base x) (g.base x)
    H : Eq (AlgebraicGeometry.Scheme.Hom.stalkMap f x) (CategoryTheory.CategoryStr …
    V : Y.Opens
    hV : Membership.mem Y.affineOpens V
    hxV : Membership.mem (↑V) (f.base x)
    hxV' : Membership.mem V (g.base x)
    U : X.Opens
    hxU : Membership.mem U x
    left✝ : AlgebraicGeometry.IsAffineOpen U
    hUV : LE.le U (Min.min ((TopologicalSpace.Opens.map f.base).obj V) ((Topologic …
    HU : Function.Injective ⇑(X.presheaf.germ U x hxU).hom
    this✝ : AlgebraicGeometry.IsAffine ↑V
    this : CategoryTheory.Mono (X.presheaf.germ U x hxU)
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.appLE f V U ⋯) (AlgebraicGeometry.Scheme.Ho …
  -/
  rw [← cancel_mono (X.presheaf.germ U x hxU)]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.e_a
    X Y : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    inst✝ : X.IsGermInjectiveAt x
    f g : Quiver.Hom X Y
    e : Eq (f.base x) (g.base x)
    H : Eq (AlgebraicGeometry.Scheme.Hom.stalkMap f x) (CategoryTheory.CategoryStr …
    V : Y.Opens
    hV : Membership.mem Y.affineOpens V
    hxV : Membership.mem (↑V) (f.base x)
    hxV' : Membership.mem V (g.base x)
    U : X.Opens
    hxU : Membership.mem U x
    left✝ : AlgebraicGeometry.IsAffineOpen U
    hUV : LE.le U (Min.min ((TopologicalSpace.Opens.map f.base).obj V) ((Topologic …
    HU : Function.Injective ⇑(X.presheaf.germ U x hxU).hom
    this✝ : AlgebraicGeometry.IsAffine ↑V
    this : CategoryTheory.Mono (X.presheaf.germ U x hxU)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.appLE f …
  -/
  simp only [Scheme.Hom.appLE, Category.assoc, X.presheaf.germ_res', ← Scheme.stalkMap_germ, H]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.e_a
    X Y : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    inst✝ : X.IsGermInjectiveAt x
    f g : Quiver.Hom X Y
    e : Eq (f.base x) (g.base x)
    H : Eq (AlgebraicGeometry.Scheme.Hom.stalkMap f x) (CategoryTheory.CategoryStr …
    V : Y.Opens
    hV : Membership.mem Y.affineOpens V
    hxV : Membership.mem (↑V) (f.base x)
    hxV' : Membership.mem V (g.base x)
    U : X.Opens
    hxU : Membership.mem U x
    left✝ : AlgebraicGeometry.IsAffineOpen U
    hUV : LE.le U (Min.min ((TopologicalSpace.Opens.map f.base).obj V) ((Topologic …
    HU : Function.Injective ⇑(X.presheaf.germ U x hxU).hom
    this✝ : AlgebraicGeometry.IsAffine ↑V
    this : CategoryTheory.Mono (X.presheaf.germ U x hxU)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Y.presheaf.germ V (f.base x) ⋯) (Cat …
  -/
  simp only [TopCat.Presheaf.germ_stalkSpecializes_assoc, Scheme.stalkMap_germ]
  /-
    🎉 no goals
  -/


/--
A variant of `spread_out_unique_of_isGermInjective`
whose condition is an equality of scheme morphisms instead of ring homomorphisms.
-/
lemma spread_out_unique_of_isGermInjective' {x : X} [X.IsGermInjectiveAt x]
    (f g : X ⟶ Y)
    (e : X.fromSpecStalk x ≫ f = X.fromSpecStalk x ≫ g) :
    ∃ (U : X.Opens), x ∈ U ∧ U.ι ≫ f = U.ι ≫ g := by
  /-
    X Y : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    inst✝ : X.IsGermInjectiveAt x
    f g : Quiver.Hom X Y
    e : Eq (CategoryTheory.CategoryStruct.comp (X.fromSpecStalk x) f) (CategoryThe …
    ⊢ Exists fun U => And (Membership.mem U x) (Eq (CategoryTheory.CategoryStruct. …
  -/
  fapply spread_out_unique_of_isGermInjective
    /-
      case e
      X Y : AlgebraicGeometry.Scheme
      x : ↑↑X.toPresheafedSpace
      inst✝ : X.IsGermInjectiveAt x
      f g : Quiver.Hom X Y
      e : Eq (CategoryTheory.CategoryStruct.comp (X.fromSpecStalk x) f) (CategoryThe …
      ⊢ Eq (f.base x) (g.base x)
    -/
  · simpa using congr(($e).base (IsLocalRing.closedPoint _))
    /-
      🎉 no goals
    -/
    /-
      case H
      X Y : AlgebraicGeometry.Scheme
      x : ↑↑X.toPresheafedSpace
      inst✝ : X.IsGermInjectiveAt x
      f g : Quiver.Hom X Y
      e : Eq (CategoryTheory.CategoryStruct.comp (X.fromSpecStalk x) f) (CategoryThe …
      ⊢ Eq (AlgebraicGeometry.Scheme.Hom.stalkMap f x) (CategoryTheory.CategoryStruc …
    -/
  · apply Spec.map_injective
    /-
      case H.a
      X Y : AlgebraicGeometry.Scheme
      x : ↑↑X.toPresheafedSpace
      inst✝ : X.IsGermInjectiveAt x
      f g : Quiver.Hom X Y
      e : Eq (CategoryTheory.CategoryStruct.comp (X.fromSpecStalk x) f) (CategoryThe …
      ⊢ Eq (AlgebraicGeometry.Spec.map (AlgebraicGeometry.Scheme.Hom.stalkMap f x))  …
    -/
    rw [← cancel_mono (Y.fromSpecStalk _)]
    /-
      case H.a
      X Y : AlgebraicGeometry.Scheme
      x : ↑↑X.toPresheafedSpace
      inst✝ : X.IsGermInjectiveAt x
      f g : Quiver.Hom X Y
      e : Eq (CategoryTheory.CategoryStruct.comp (X.fromSpecStalk x) f) (CategoryThe …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Algebrai …
    -/
    simpa [Scheme.Spec_map_stalkSpecializes_fromSpecStalk]
    /-
      🎉 no goals
    -/


lemma exists_lift_of_germInjective_aux {U : X.Opens} {x : X} (hxU)
    (φ : A ⟶ X.presheaf.stalk x) (φRA : R ⟶ A) (φRX : R ⟶ Γ(X, U))
    (hφRA : RingHom.FiniteType φRA.hom)
    (e : φRA ≫ φ = φRX ≫ X.presheaf.germ U x hxU) :
    ∃ (V : X.Opens) (hxV : x ∈ V),
      V ≤ U ∧ RingHom.range φ.hom ≤ RingHom.range (X.presheaf.germ V x hxV).hom := by
  /-
    X : AlgebraicGeometry.Scheme
    R A : CommRingCat
    U : X.Opens
    x : ↑↑X.toPresheafedSpace
    hxU : Membership.mem U x
    φ : Quiver.Hom A (X.presheaf.stalk x)
    φRA : Quiver.Hom R A
    φRX : Quiver.Hom R (X.presheaf.obj { unop := U })
    hφRA : φRA.hom.FiniteType
    e : Eq (CategoryTheory.CategoryStruct.comp φRA φ) (CategoryTheory.CategoryStru …
    ⊢ Exists fun V => Exists fun hxV => And (LE.le V U) (LE.le φ.hom.range (X.pres …
  -/
  letI := φRA.hom.toAlgebra
  /-
    X : AlgebraicGeometry.Scheme
    R A : CommRingCat
    U : X.Opens
    x : ↑↑X.toPresheafedSpace
    hxU : Membership.mem U x
    φ : Quiver.Hom A (X.presheaf.stalk x)
    φRA : Quiver.Hom R A
    φRX : Quiver.Hom R (X.presheaf.obj { unop := U })
    hφRA : φRA.hom.FiniteType
    e : Eq (CategoryTheory.CategoryStruct.comp φRA φ) (CategoryTheory.CategoryStru …
    this : Algebra ↑R ↑A := φRA.hom.toAlgebra
    ⊢ Exists fun V => Exists fun hxV => And (LE.le V U) (LE.le φ.hom.range (X.pres …
  -/
  obtain ⟨s, hs⟩ := hφRA
  /-
    case mk.intro
    X : AlgebraicGeometry.Scheme
    R A : CommRingCat
    U : X.Opens
    x : ↑↑X.toPresheafedSpace
    hxU : Membership.mem U x
    φ : Quiver.Hom A (X.presheaf.stalk x)
    φRA : Quiver.Hom R A
    φRX : Quiver.Hom R (X.presheaf.obj { unop := U })
    e : Eq (CategoryTheory.CategoryStruct.comp φRA φ) (CategoryTheory.CategoryStru …
    this : Algebra ↑R ↑A := φRA.hom.toAlgebra
    s : Finset ↑A
    hs : Eq (Algebra.adjoin ↑R ↑s) Top.top
    ⊢ Exists fun V => Exists fun hxV => And (LE.le V U) (LE.le φ.hom.range (X.pres …
  -/
  choose W hxW f hf using fun t ↦ X.presheaf.germ_exist x (φ t)
  have H : x ∈ s.inf W ⊓ U := by
    rw [← SetLike.mem_coe, TopologicalSpace.Opens.coe_inf, TopologicalSpace.Opens.coe_finset_inf]
    exact ⟨by simpa using fun x _ ↦ hxW x, hxU⟩
  /-
    case mk.intro
    X : AlgebraicGeometry.Scheme
    R A : CommRingCat
    U : X.Opens
    x : ↑↑X.toPresheafedSpace
    hxU : Membership.mem U x
    φ : Quiver.Hom A (X.presheaf.stalk x)
    φRA : Quiver.Hom R A
    φRX : Quiver.Hom R (X.presheaf.obj { unop := U })
    e : Eq (CategoryTheory.CategoryStruct.comp φRA φ) (CategoryTheory.CategoryStru …
    this : Algebra ↑R ↑A := φRA.hom.toAlgebra
    s : Finset ↑A
    hs : Eq (Algebra.adjoin ↑R ↑s) Top.top
    W : ↑A → TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hxW : ∀ (t : ↑A), Membership.mem (W t) x
    f : (t : ↑A) → (CategoryTheory.forget CommRingCat).obj (X.presheaf.obj { unop  …
    hf : ∀ (t : ↑A), Eq ((X.presheaf.germ (W t) x ⋯) (f t)) (φ.hom t)
    H : Membership.mem (Min.min (s.inf W) U) x
    ⊢ Exists fun V => Exists fun hxV => And (LE.le V U) (LE.le φ.hom.range (X.pres …
  -/
  refine ⟨s.inf W ⊓ U, H, inf_le_right, ?_⟩
  /-
    case mk.intro
    X : AlgebraicGeometry.Scheme
    R A : CommRingCat
    U : X.Opens
    x : ↑↑X.toPresheafedSpace
    hxU : Membership.mem U x
    φ : Quiver.Hom A (X.presheaf.stalk x)
    φRA : Quiver.Hom R A
    φRX : Quiver.Hom R (X.presheaf.obj { unop := U })
    e : Eq (CategoryTheory.CategoryStruct.comp φRA φ) (CategoryTheory.CategoryStru …
    this : Algebra ↑R ↑A := φRA.hom.toAlgebra
    s : Finset ↑A
    hs : Eq (Algebra.adjoin ↑R ↑s) Top.top
    W : ↑A → TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hxW : ∀ (t : ↑A), Membership.mem (W t) x
    f : (t : ↑A) → (CategoryTheory.forget CommRingCat).obj (X.presheaf.obj { unop  …
    hf : ∀ (t : ↑A), Eq ((X.presheaf.germ (W t) x ⋯) (f t)) (φ.hom t)
    H : Membership.mem (Min.min (s.inf W) U) x
    ⊢ LE.le φ.hom.range (X.presheaf.germ (Min.min (s.inf W) U) x H).hom.range
  -/
  letI := φRX.hom.toAlgebra
  /-
    case mk.intro
    X : AlgebraicGeometry.Scheme
    R A : CommRingCat
    U : X.Opens
    x : ↑↑X.toPresheafedSpace
    hxU : Membership.mem U x
    φ : Quiver.Hom A (X.presheaf.stalk x)
    φRA : Quiver.Hom R A
    φRX : Quiver.Hom R (X.presheaf.obj { unop := U })
    e : Eq (CategoryTheory.CategoryStruct.comp φRA φ) (CategoryTheory.CategoryStru …
    this✝ : Algebra ↑R ↑A := φRA.hom.toAlgebra
    s : Finset ↑A
    hs : Eq (Algebra.adjoin ↑R ↑s) Top.top
    W : ↑A → TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hxW : ∀ (t : ↑A), Membership.mem (W t) x
    f : (t : ↑A) → (CategoryTheory.forget CommRingCat).obj (X.presheaf.obj { unop  …
    hf : ∀ (t : ↑A), Eq ((X.presheaf.germ (W t) x ⋯) (f t)) (φ.hom t)
    H : Membership.mem (Min.min (s.inf W) U) x
    this : Algebra ↑R ↑(X.presheaf.obj { unop := U }) := φRX.hom.toAlgebra
    ⊢ LE.le φ.hom.range (X.presheaf.germ (Min.min (s.inf W) U) x H).hom.range
  -/
  letI := (φRX ≫ X.presheaf.germ U x hxU).hom.toAlgebra
  /-
    case mk.intro
    X : AlgebraicGeometry.Scheme
    R A : CommRingCat
    U : X.Opens
    x : ↑↑X.toPresheafedSpace
    hxU : Membership.mem U x
    φ : Quiver.Hom A (X.presheaf.stalk x)
    φRA : Quiver.Hom R A
    φRX : Quiver.Hom R (X.presheaf.obj { unop := U })
    e : Eq (CategoryTheory.CategoryStruct.comp φRA φ) (CategoryTheory.CategoryStru …
    this✝¹ : Algebra ↑R ↑A := φRA.hom.toAlgebra
    s : Finset ↑A
    hs : Eq (Algebra.adjoin ↑R ↑s) Top.top
    W : ↑A → TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hxW : ∀ (t : ↑A), Membership.mem (W t) x
    f : (t : ↑A) → (CategoryTheory.forget CommRingCat).obj (X.presheaf.obj { unop  …
    hf : ∀ (t : ↑A), Eq ((X.presheaf.germ (W t) x ⋯) (f t)) (φ.hom t)
    H : Membership.mem (Min.min (s.inf W) U) x
    this✝ : Algebra ↑R ↑(X.presheaf.obj { unop := U }) := φRX.hom.toAlgebra
    this : Algebra ↑R ↑(X.presheaf.stalk x) := (CategoryTheory.CategoryStruct.comp …
    ⊢ LE.le φ.hom.range (X.presheaf.germ (Min.min (s.inf W) U) x H).hom.range
  -/
  letI := (φRX ≫ X.presheaf.map (homOfLE (inf_le_right (a := s.inf W))).op).hom.toAlgebra
  let φ' : A →ₐ[R] X.presheaf.stalk x :=
    { φ.hom with commutes' := DFunLike.congr_fun (congr_arg CommRingCat.Hom.hom e) }
  let ψ : Γ(X, s.inf W ⊓ U) →ₐ[R] X.presheaf.stalk x :=
    { (X.presheaf.germ _ x H).hom with commutes' := fun x ↦ X.presheaf.germ_res_apply _ _ _ _ }
  /-
    case mk.intro
    X : AlgebraicGeometry.Scheme
    R A : CommRingCat
    U : X.Opens
    x : ↑↑X.toPresheafedSpace
    hxU : Membership.mem U x
    φ : Quiver.Hom A (X.presheaf.stalk x)
    φRA : Quiver.Hom R A
    φRX : Quiver.Hom R (X.presheaf.obj { unop := U })
    e : Eq (CategoryTheory.CategoryStruct.comp φRA φ) (CategoryTheory.CategoryStru …
    this✝² : Algebra ↑R ↑A := φRA.hom.toAlgebra
    s : Finset ↑A
    hs : Eq (Algebra.adjoin ↑R ↑s) Top.top
    W : ↑A → TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hxW : ∀ (t : ↑A), Membership.mem (W t) x
    f : (t : ↑A) → (CategoryTheory.forget CommRingCat).obj (X.presheaf.obj { unop  …
    hf : ∀ (t : ↑A), Eq ((X.presheaf.germ (W t) x ⋯) (f t)) (φ.hom t)
    H : Membership.mem (Min.min (s.inf W) U) x
    this✝¹ : Algebra ↑R ↑(X.presheaf.obj { unop := U }) := φRX.hom.toAlgebra
    this✝ : Algebra ↑R ↑(X.presheaf.stalk x) := (CategoryTheory.CategoryStruct.com …
    this : Algebra ↑R ↑(X.presheaf.obj { unop := Min.min (s.inf W) U }) := (Catego …
    φ' : AlgHom ↑R ↑A ↑(X.presheaf.stalk x) :=
      let __src := φ.hom;
      { toRingHom := __src, commutes' := ⋯ }
    ψ : AlgHom ↑R ↑(X.presheaf.obj { unop := Min.min (s.inf W) U }) ↑(X.presheaf.s …
      let __src := (X.presheaf.germ (Min.min (s.inf W) U) x H).hom;
      { toRingHom := __src, commutes' := ⋯ }
    ⊢ LE.le φ.hom.range (X.presheaf.germ (Min.min (s.inf W) U) x H).hom.range
  -/
  show AlgHom.range φ' ≤ AlgHom.range ψ
  /-
    case mk.intro
    X : AlgebraicGeometry.Scheme
    R A : CommRingCat
    U : X.Opens
    x : ↑↑X.toPresheafedSpace
    hxU : Membership.mem U x
    φ : Quiver.Hom A (X.presheaf.stalk x)
    φRA : Quiver.Hom R A
    φRX : Quiver.Hom R (X.presheaf.obj { unop := U })
    e : Eq (CategoryTheory.CategoryStruct.comp φRA φ) (CategoryTheory.CategoryStru …
    this✝² : Algebra ↑R ↑A := φRA.hom.toAlgebra
    s : Finset ↑A
    hs : Eq (Algebra.adjoin ↑R ↑s) Top.top
    W : ↑A → TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hxW : ∀ (t : ↑A), Membership.mem (W t) x
    f : (t : ↑A) → (CategoryTheory.forget CommRingCat).obj (X.presheaf.obj { unop  …
    hf : ∀ (t : ↑A), Eq ((X.presheaf.germ (W t) x ⋯) (f t)) (φ.hom t)
    H : Membership.mem (Min.min (s.inf W) U) x
    this✝¹ : Algebra ↑R ↑(X.presheaf.obj { unop := U }) := φRX.hom.toAlgebra
    this✝ : Algebra ↑R ↑(X.presheaf.stalk x) := (CategoryTheory.CategoryStruct.com …
    this : Algebra ↑R ↑(X.presheaf.obj { unop := Min.min (s.inf W) U }) := (Catego …
    φ' : AlgHom ↑R ↑A ↑(X.presheaf.stalk x) :=
      let __src := φ.hom;
      { toRingHom := __src, commutes' := ⋯ }
    ψ : AlgHom ↑R ↑(X.presheaf.obj { unop := Min.min (s.inf W) U }) ↑(X.presheaf.s …
      let __src := (X.presheaf.germ (Min.min (s.inf W) U) x H).hom;
      { toRingHom := __src, commutes' := ⋯ }
    ⊢ LE.le φ'.range ψ.range
  -/
  rw [← Algebra.map_top, ← hs, AlgHom.map_adjoin, Algebra.adjoin_le_iff]
  /-
    case mk.intro
    X : AlgebraicGeometry.Scheme
    R A : CommRingCat
    U : X.Opens
    x : ↑↑X.toPresheafedSpace
    hxU : Membership.mem U x
    φ : Quiver.Hom A (X.presheaf.stalk x)
    φRA : Quiver.Hom R A
    φRX : Quiver.Hom R (X.presheaf.obj { unop := U })
    e : Eq (CategoryTheory.CategoryStruct.comp φRA φ) (CategoryTheory.CategoryStru …
    this✝² : Algebra ↑R ↑A := φRA.hom.toAlgebra
    s : Finset ↑A
    hs : Eq (Algebra.adjoin ↑R ↑s) Top.top
    W : ↑A → TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hxW : ∀ (t : ↑A), Membership.mem (W t) x
    f : (t : ↑A) → (CategoryTheory.forget CommRingCat).obj (X.presheaf.obj { unop  …
    hf : ∀ (t : ↑A), Eq ((X.presheaf.germ (W t) x ⋯) (f t)) (φ.hom t)
    H : Membership.mem (Min.min (s.inf W) U) x
    this✝¹ : Algebra ↑R ↑(X.presheaf.obj { unop := U }) := φRX.hom.toAlgebra
    this✝ : Algebra ↑R ↑(X.presheaf.stalk x) := (CategoryTheory.CategoryStruct.com …
    this : Algebra ↑R ↑(X.presheaf.obj { unop := Min.min (s.inf W) U }) := (Catego …
    φ' : AlgHom ↑R ↑A ↑(X.presheaf.stalk x) :=
      let __src := φ.hom;
      { toRingHom := __src, commutes' := ⋯ }
    ψ : AlgHom ↑R ↑(X.presheaf.obj { unop := Min.min (s.inf W) U }) ↑(X.presheaf.s …
      let __src := (X.presheaf.germ (Min.min (s.inf W) U) x H).hom;
      { toRingHom := __src, commutes' := ⋯ }
    ⊢ HasSubset.Subset (Set.image ⇑φ' ↑s) ↑ψ.range
  -/
  rintro _ ⟨i, hi, rfl : φ i = _⟩
  /-
    case mk.intro.intro.intro
    X : AlgebraicGeometry.Scheme
    R A : CommRingCat
    U : X.Opens
    x : ↑↑X.toPresheafedSpace
    hxU : Membership.mem U x
    φ : Quiver.Hom A (X.presheaf.stalk x)
    φRA : Quiver.Hom R A
    φRX : Quiver.Hom R (X.presheaf.obj { unop := U })
    e : Eq (CategoryTheory.CategoryStruct.comp φRA φ) (CategoryTheory.CategoryStru …
    this✝² : Algebra ↑R ↑A := φRA.hom.toAlgebra
    s : Finset ↑A
    hs : Eq (Algebra.adjoin ↑R ↑s) Top.top
    W : ↑A → TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hxW : ∀ (t : ↑A), Membership.mem (W t) x
    f : (t : ↑A) → (CategoryTheory.forget CommRingCat).obj (X.presheaf.obj { unop  …
    hf : ∀ (t : ↑A), Eq ((X.presheaf.germ (W t) x ⋯) (f t)) (φ.hom t)
    H : Membership.mem (Min.min (s.inf W) U) x
    this✝¹ : Algebra ↑R ↑(X.presheaf.obj { unop := U }) := φRX.hom.toAlgebra
    this✝ : Algebra ↑R ↑(X.presheaf.stalk x) := (CategoryTheory.CategoryStruct.com …
    this : Algebra ↑R ↑(X.presheaf.obj { unop := Min.min (s.inf W) U }) := (Catego …
    φ' : AlgHom ↑R ↑A ↑(X.presheaf.stalk x) :=
      let __src := φ.hom;
      { toRingHom := __src, commutes' := ⋯ }
    ψ : AlgHom ↑R ↑(X.presheaf.obj { unop := Min.min (s.inf W) U }) ↑(X.presheaf.s …
      let __src := (X.presheaf.germ (Min.min (s.inf W) U) x H).hom;
      { toRingHom := __src, commutes' := ⋯ }
    i : ↑A
    hi : Membership.mem (↑s) i
    ⊢ Membership.mem (↑ψ.range) (φ.hom i)
  -/
  refine ⟨X.presheaf.map (homOfLE (inf_le_left.trans (Finset.inf_le hi))).op (f i), ?_⟩
  /-
    case mk.intro.intro.intro
    X : AlgebraicGeometry.Scheme
    R A : CommRingCat
    U : X.Opens
    x : ↑↑X.toPresheafedSpace
    hxU : Membership.mem U x
    φ : Quiver.Hom A (X.presheaf.stalk x)
    φRA : Quiver.Hom R A
    φRX : Quiver.Hom R (X.presheaf.obj { unop := U })
    e : Eq (CategoryTheory.CategoryStruct.comp φRA φ) (CategoryTheory.CategoryStru …
    this✝² : Algebra ↑R ↑A := φRA.hom.toAlgebra
    s : Finset ↑A
    hs : Eq (Algebra.adjoin ↑R ↑s) Top.top
    W : ↑A → TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hxW : ∀ (t : ↑A), Membership.mem (W t) x
    f : (t : ↑A) → (CategoryTheory.forget CommRingCat).obj (X.presheaf.obj { unop  …
    hf : ∀ (t : ↑A), Eq ((X.presheaf.germ (W t) x ⋯) (f t)) (φ.hom t)
    H : Membership.mem (Min.min (s.inf W) U) x
    this✝¹ : Algebra ↑R ↑(X.presheaf.obj { unop := U }) := φRX.hom.toAlgebra
    this✝ : Algebra ↑R ↑(X.presheaf.stalk x) := (CategoryTheory.CategoryStruct.com …
    this : Algebra ↑R ↑(X.presheaf.obj { unop := Min.min (s.inf W) U }) := (Catego …
    φ' : AlgHom ↑R ↑A ↑(X.presheaf.stalk x) :=
      let __src := φ.hom;
      { toRingHom := __src, commutes' := ⋯ }
    ψ : AlgHom ↑R ↑(X.presheaf.obj { unop := Min.min (s.inf W) U }) ↑(X.presheaf.s …
      let __src := (X.presheaf.germ (Min.min (s.inf W) U) x H).hom;
      { toRingHom := __src, commutes' := ⋯ }
    i : ↑A
    hi : Membership.mem (↑s) i
    ⊢ Eq (ψ.toRingHom ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom (f i)))  …
  -/
  exact (X.presheaf.germ_res_apply _ _ _ _).trans (hf _)
  /-
    🎉 no goals
  -/


/--
Suppose `X` is a scheme, `x : X` such that the germ map at `x` is (locally) injective,
and `U` is a neighborhood of `x`.
Given a commutative diagram of `CommRingCat`
```
R ⟶ Γ(X, U)
↓    ↓
A ⟶ 𝒪_{X, x}
```
such that `R` is of finite type over `A`, we may lift `A ⟶ 𝒪_{X, x}` to some `A ⟶ Γ(X, V)`.
-/
lemma exists_lift_of_germInjective {x : X} [X.IsGermInjectiveAt x] {U : X.Opens} (hxU : x ∈ U)
    (φ : A ⟶ X.presheaf.stalk x) (φRA : R ⟶ A) (φRX : R ⟶ Γ(X, U))
    (hφRA : RingHom.FiniteType φRA.hom)
    (e : φRA ≫ φ = φRX ≫ X.presheaf.germ U x hxU) :
    ∃ (V : X.Opens) (hxV : x ∈ V) (φ' : A ⟶ Γ(X, V)) (i : V ≤ U), IsAffineOpen V ∧
      φ = φ' ≫ X.presheaf.germ V x hxV ∧ φRX ≫ X.presheaf.map i.hom.op = φRA ≫ φ' := by
  /-
    X : AlgebraicGeometry.Scheme
    R A : CommRingCat
    x : ↑↑X.toPresheafedSpace
    inst✝ : X.IsGermInjectiveAt x
    U : X.Opens
    hxU : Membership.mem U x
    φ : Quiver.Hom A (X.presheaf.stalk x)
    φRA : Quiver.Hom R A
    φRX : Quiver.Hom R (X.presheaf.obj { unop := U })
    hφRA : φRA.hom.FiniteType
    e : Eq (CategoryTheory.CategoryStruct.comp φRA φ) (CategoryTheory.CategoryStru …
    ⊢ Exists fun V => Exists fun hxV => Exists fun φ' => Exists fun i => And (Alge …
  -/
  obtain ⟨V, hxV, iVU, hV⟩ := exists_lift_of_germInjective_aux hxU φ φRA φRX hφRA e
  /-
    case intro.intro.intro
    X : AlgebraicGeometry.Scheme
    R A : CommRingCat
    x : ↑↑X.toPresheafedSpace
    inst✝ : X.IsGermInjectiveAt x
    U : X.Opens
    hxU : Membership.mem U x
    φ : Quiver.Hom A (X.presheaf.stalk x)
    φRA : Quiver.Hom R A
    φRX : Quiver.Hom R (X.presheaf.obj { unop := U })
    hφRA : φRA.hom.FiniteType
    e : Eq (CategoryTheory.CategoryStruct.comp φRA φ) (CategoryTheory.CategoryStru …
    V : X.Opens
    hxV : Membership.mem V x
    iVU : LE.le V U
    hV : LE.le φ.hom.range (X.presheaf.germ V x hxV).hom.range
    ⊢ Exists fun V => Exists fun hxV => Exists fun φ' => Exists fun i => And (Alge …
  -/
  obtain ⟨V', hxV', hV', iV'V, H⟩ := X.exists_le_and_germ_injective x V hxV
  /-
    case intro.intro.intro.intro.intro.intro.intro
    X : AlgebraicGeometry.Scheme
    R A : CommRingCat
    x : ↑↑X.toPresheafedSpace
    inst✝ : X.IsGermInjectiveAt x
    U : X.Opens
    hxU : Membership.mem U x
    φ : Quiver.Hom A (X.presheaf.stalk x)
    φRA : Quiver.Hom R A
    φRX : Quiver.Hom R (X.presheaf.obj { unop := U })
    hφRA : φRA.hom.FiniteType
    e : Eq (CategoryTheory.CategoryStruct.comp φRA φ) (CategoryTheory.CategoryStru …
    V : X.Opens
    hxV : Membership.mem V x
    iVU : LE.le V U
    hV : LE.le φ.hom.range (X.presheaf.germ V x hxV).hom.range
    V' : X.Opens
    hxV' : Membership.mem V' x
    hV' : AlgebraicGeometry.IsAffineOpen V'
    iV'V : LE.le V' V
    H : Function.Injective ⇑(X.presheaf.germ V' x hxV').hom
    ⊢ Exists fun V => Exists fun hxV => Exists fun φ' => Exists fun i => And (Alge …
  -/
  let f := X.presheaf.germ V' x hxV'
  have hf' : RingHom.range (X.presheaf.germ V x hxV).hom ≤ RingHom.range f.hom := by
    rw [← X.presheaf.germ_res iV'V.hom _ hxV']
    exact Set.range_comp_subset_range (X.presheaf.map iV'V.hom.op) f
  /-
    case intro.intro.intro.intro.intro.intro.intro
    X : AlgebraicGeometry.Scheme
    R A : CommRingCat
    x : ↑↑X.toPresheafedSpace
    inst✝ : X.IsGermInjectiveAt x
    U : X.Opens
    hxU : Membership.mem U x
    φ : Quiver.Hom A (X.presheaf.stalk x)
    φRA : Quiver.Hom R A
    φRX : Quiver.Hom R (X.presheaf.obj { unop := U })
    hφRA : φRA.hom.FiniteType
    e : Eq (CategoryTheory.CategoryStruct.comp φRA φ) (CategoryTheory.CategoryStru …
    V : X.Opens
    hxV : Membership.mem V x
    iVU : LE.le V U
    hV : LE.le φ.hom.range (X.presheaf.germ V x hxV).hom.range
    V' : X.Opens
    hxV' : Membership.mem V' x
    hV' : AlgebraicGeometry.IsAffineOpen V'
    iV'V : LE.le V' V
    H : Function.Injective ⇑(X.presheaf.germ V' x hxV').hom
    f : Quiver.Hom (X.presheaf.obj { unop := V' }) (X.presheaf.stalk x) := X.presh …
    hf' : LE.le (X.presheaf.germ V x hxV).hom.range f.hom.range
    ⊢ Exists fun V => Exists fun hxV => Exists fun φ' => Exists fun i => And (Alge …
  -/
  let e := RingEquiv.ofLeftInverse H.hasLeftInverse.choose_spec
  refine ⟨V', hxV', CommRingCat.ofHom (e.symm.toRingHom.comp
    (φ.hom.codRestrict _ (fun x ↦ hf' (hV ⟨x, rfl⟩)))), iV'V.trans iVU, hV', ?_, ?_⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.refine_1
      X : AlgebraicGeometry.Scheme
      R A : CommRingCat
      x : ↑↑X.toPresheafedSpace
      inst✝ : X.IsGermInjectiveAt x
      U : X.Opens
      hxU : Membership.mem U x
      φ : Quiver.Hom A (X.presheaf.stalk x)
      φRA : Quiver.Hom R A
      φRX : Quiver.Hom R (X.presheaf.obj { unop := U })
      hφRA : φRA.hom.FiniteType
      e✝ : Eq (CategoryTheory.CategoryStruct.comp φRA φ) (CategoryTheory.CategoryStr …
      V : X.Opens
      hxV : Membership.mem V x
      iVU : LE.le V U
      hV : LE.le φ.hom.range (X.presheaf.germ V x hxV).hom.range
      V' : X.Opens
      hxV' : Membership.mem V' x
      hV' : AlgebraicGeometry.IsAffineOpen V'
      iV'V : LE.le V' V
      H : Function.Injective ⇑(X.presheaf.germ V' x hxV').hom
      f : Quiver.Hom (X.presheaf.obj { unop := V' }) (X.presheaf.stalk x) := X.presh …
      hf' : LE.le (X.presheaf.germ V x hxV).hom.range f.hom.range
      e : RingEquiv (↑(X.presheaf.obj { unop := V' })) (Subtype fun x_1 => Membershi …
      ⊢ Eq φ (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom (e.symm.toRingHo …
    -/
  · ext a
    /-
      case intro.intro.intro.intro.intro.intro.intro.refine_1.hf.a
      X : AlgebraicGeometry.Scheme
      R A : CommRingCat
      x : ↑↑X.toPresheafedSpace
      inst✝ : X.IsGermInjectiveAt x
      U : X.Opens
      hxU : Membership.mem U x
      φ : Quiver.Hom A (X.presheaf.stalk x)
      φRA : Quiver.Hom R A
      φRX : Quiver.Hom R (X.presheaf.obj { unop := U })
      hφRA : φRA.hom.FiniteType
      e✝ : Eq (CategoryTheory.CategoryStruct.comp φRA φ) (CategoryTheory.CategoryStr …
      V : X.Opens
      hxV : Membership.mem V x
      iVU : LE.le V U
      hV : LE.le φ.hom.range (X.presheaf.germ V x hxV).hom.range
      V' : X.Opens
      hxV' : Membership.mem V' x
      hV' : AlgebraicGeometry.IsAffineOpen V'
      iV'V : LE.le V' V
      H : Function.Injective ⇑(X.presheaf.germ V' x hxV').hom
      f : Quiver.Hom (X.presheaf.obj { unop := V' }) (X.presheaf.stalk x) := X.presh …
      hf' : LE.le (X.presheaf.germ V x hxV).hom.range f.hom.range
      e : RingEquiv (↑(X.presheaf.obj { unop := V' })) (Subtype fun x_1 => Membershi …
      a : ↑A
      ⊢ Eq (φ.hom a) ((CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom (e.symm …
    -/
    show φ a = (e (e.symm _)).1
    /-
      case intro.intro.intro.intro.intro.intro.intro.refine_1.hf.a
      X : AlgebraicGeometry.Scheme
      R A : CommRingCat
      x : ↑↑X.toPresheafedSpace
      inst✝ : X.IsGermInjectiveAt x
      U : X.Opens
      hxU : Membership.mem U x
      φ : Quiver.Hom A (X.presheaf.stalk x)
      φRA : Quiver.Hom R A
      φRX : Quiver.Hom R (X.presheaf.obj { unop := U })
      hφRA : φRA.hom.FiniteType
      e✝ : Eq (CategoryTheory.CategoryStruct.comp φRA φ) (CategoryTheory.CategoryStr …
      V : X.Opens
      hxV : Membership.mem V x
      iVU : LE.le V U
      hV : LE.le φ.hom.range (X.presheaf.germ V x hxV).hom.range
      V' : X.Opens
      hxV' : Membership.mem V' x
      hV' : AlgebraicGeometry.IsAffineOpen V'
      iV'V : LE.le V' V
      H : Function.Injective ⇑(X.presheaf.germ V' x hxV').hom
      f : Quiver.Hom (X.presheaf.obj { unop := V' }) (X.presheaf.stalk x) := X.presh …
      hf' : LE.le (X.presheaf.germ V x hxV).hom.range f.hom.range
      e : RingEquiv (↑(X.presheaf.obj { unop := V' })) (Subtype fun x_1 => Membershi …
      a : ↑A
      ⊢ Eq (φ.hom a) ↑(e (e.symm ((φ.hom.codRestrict (X.presheaf.germ V' x hxV').hom …
    -/
    simp only [RingEquiv.apply_symm_apply]
    /-
      case intro.intro.intro.intro.intro.intro.intro.refine_1.hf.a
      X : AlgebraicGeometry.Scheme
      R A : CommRingCat
      x : ↑↑X.toPresheafedSpace
      inst✝ : X.IsGermInjectiveAt x
      U : X.Opens
      hxU : Membership.mem U x
      φ : Quiver.Hom A (X.presheaf.stalk x)
      φRA : Quiver.Hom R A
      φRX : Quiver.Hom R (X.presheaf.obj { unop := U })
      hφRA : φRA.hom.FiniteType
      e✝ : Eq (CategoryTheory.CategoryStruct.comp φRA φ) (CategoryTheory.CategoryStr …
      V : X.Opens
      hxV : Membership.mem V x
      iVU : LE.le V U
      hV : LE.le φ.hom.range (X.presheaf.germ V x hxV).hom.range
      V' : X.Opens
      hxV' : Membership.mem V' x
      hV' : AlgebraicGeometry.IsAffineOpen V'
      iV'V : LE.le V' V
      H : Function.Injective ⇑(X.presheaf.germ V' x hxV').hom
      f : Quiver.Hom (X.presheaf.obj { unop := V' }) (X.presheaf.stalk x) := X.presh …
      hf' : LE.le (X.presheaf.germ V x hxV).hom.range f.hom.range
      e : RingEquiv (↑(X.presheaf.obj { unop := V' })) (Subtype fun x_1 => Membershi …
      a : ↑A
      ⊢ Eq (φ.hom a) ↑((φ.hom.codRestrict (X.presheaf.germ V' x hxV').hom.range ⋯) a)
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.refine_2
      X : AlgebraicGeometry.Scheme
      R A : CommRingCat
      x : ↑↑X.toPresheafedSpace
      inst✝ : X.IsGermInjectiveAt x
      U : X.Opens
      hxU : Membership.mem U x
      φ : Quiver.Hom A (X.presheaf.stalk x)
      φRA : Quiver.Hom R A
      φRX : Quiver.Hom R (X.presheaf.obj { unop := U })
      hφRA : φRA.hom.FiniteType
      e✝ : Eq (CategoryTheory.CategoryStruct.comp φRA φ) (CategoryTheory.CategoryStr …
      V : X.Opens
      hxV : Membership.mem V x
      iVU : LE.le V U
      hV : LE.le φ.hom.range (X.presheaf.germ V x hxV).hom.range
      V' : X.Opens
      hxV' : Membership.mem V' x
      hV' : AlgebraicGeometry.IsAffineOpen V'
      iV'V : LE.le V' V
      H : Function.Injective ⇑(X.presheaf.germ V' x hxV').hom
      f : Quiver.Hom (X.presheaf.obj { unop := V' }) (X.presheaf.stalk x) := X.presh …
      hf' : LE.le (X.presheaf.germ V x hxV).hom.range f.hom.range
      e : RingEquiv (↑(X.presheaf.obj { unop := V' })) (Subtype fun x_1 => Membershi …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φRX (X.presheaf.map ⋯.hom.op)) (Categ …
    -/
  · ext a
    /-
      case intro.intro.intro.intro.intro.intro.intro.refine_2.hf.a
      X : AlgebraicGeometry.Scheme
      R A : CommRingCat
      x : ↑↑X.toPresheafedSpace
      inst✝ : X.IsGermInjectiveAt x
      U : X.Opens
      hxU : Membership.mem U x
      φ : Quiver.Hom A (X.presheaf.stalk x)
      φRA : Quiver.Hom R A
      φRX : Quiver.Hom R (X.presheaf.obj { unop := U })
      hφRA : φRA.hom.FiniteType
      e✝ : Eq (CategoryTheory.CategoryStruct.comp φRA φ) (CategoryTheory.CategoryStr …
      V : X.Opens
      hxV : Membership.mem V x
      iVU : LE.le V U
      hV : LE.le φ.hom.range (X.presheaf.germ V x hxV).hom.range
      V' : X.Opens
      hxV' : Membership.mem V' x
      hV' : AlgebraicGeometry.IsAffineOpen V'
      iV'V : LE.le V' V
      H : Function.Injective ⇑(X.presheaf.germ V' x hxV').hom
      f : Quiver.Hom (X.presheaf.obj { unop := V' }) (X.presheaf.stalk x) := X.presh …
      hf' : LE.le (X.presheaf.germ V x hxV).hom.range f.hom.range
      e : RingEquiv (↑(X.presheaf.obj { unop := V' })) (Subtype fun x_1 => Membershi …
      a : ↑R
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp φRX (X.presheaf.map ⋯.hom.op)).hom a …
    -/
    apply e.injective
    /-
      case intro.intro.intro.intro.intro.intro.intro.refine_2.hf.a.a
      X : AlgebraicGeometry.Scheme
      R A : CommRingCat
      x : ↑↑X.toPresheafedSpace
      inst✝ : X.IsGermInjectiveAt x
      U : X.Opens
      hxU : Membership.mem U x
      φ : Quiver.Hom A (X.presheaf.stalk x)
      φRA : Quiver.Hom R A
      φRX : Quiver.Hom R (X.presheaf.obj { unop := U })
      hφRA : φRA.hom.FiniteType
      e✝ : Eq (CategoryTheory.CategoryStruct.comp φRA φ) (CategoryTheory.CategoryStr …
      V : X.Opens
      hxV : Membership.mem V x
      iVU : LE.le V U
      hV : LE.le φ.hom.range (X.presheaf.germ V x hxV).hom.range
      V' : X.Opens
      hxV' : Membership.mem V' x
      hV' : AlgebraicGeometry.IsAffineOpen V'
      iV'V : LE.le V' V
      H : Function.Injective ⇑(X.presheaf.germ V' x hxV').hom
      f : Quiver.Hom (X.presheaf.obj { unop := V' }) (X.presheaf.stalk x) := X.presh …
      hf' : LE.le (X.presheaf.germ V x hxV).hom.range f.hom.range
      e : RingEquiv (↑(X.presheaf.obj { unop := V' })) (Subtype fun x_1 => Membershi …
      a : ↑R
      ⊢ Eq (e ((CategoryTheory.CategoryStruct.comp φRX (X.presheaf.map ⋯.hom.op)).ho …
    -/
    show e _ = e (e.symm _)
    /-
      case intro.intro.intro.intro.intro.intro.intro.refine_2.hf.a.a
      X : AlgebraicGeometry.Scheme
      R A : CommRingCat
      x : ↑↑X.toPresheafedSpace
      inst✝ : X.IsGermInjectiveAt x
      U : X.Opens
      hxU : Membership.mem U x
      φ : Quiver.Hom A (X.presheaf.stalk x)
      φRA : Quiver.Hom R A
      φRX : Quiver.Hom R (X.presheaf.obj { unop := U })
      hφRA : φRA.hom.FiniteType
      e✝ : Eq (CategoryTheory.CategoryStruct.comp φRA φ) (CategoryTheory.CategoryStr …
      V : X.Opens
      hxV : Membership.mem V x
      iVU : LE.le V U
      hV : LE.le φ.hom.range (X.presheaf.germ V x hxV).hom.range
      V' : X.Opens
      hxV' : Membership.mem V' x
      hV' : AlgebraicGeometry.IsAffineOpen V'
      iV'V : LE.le V' V
      H : Function.Injective ⇑(X.presheaf.germ V' x hxV').hom
      f : Quiver.Hom (X.presheaf.obj { unop := V' }) (X.presheaf.stalk x) := X.presh …
      hf' : LE.le (X.presheaf.germ V x hxV).hom.range f.hom.range
      e : RingEquiv (↑(X.presheaf.obj { unop := V' })) (Subtype fun x_1 => Membershi …
      a : ↑R
      ⊢ Eq (e ((CategoryTheory.CategoryStruct.comp φRX (X.presheaf.map ⋯.hom.op)).ho …
    -/
    rw [RingEquiv.apply_symm_apply]
    /-
      case intro.intro.intro.intro.intro.intro.intro.refine_2.hf.a.a
      X : AlgebraicGeometry.Scheme
      R A : CommRingCat
      x : ↑↑X.toPresheafedSpace
      inst✝ : X.IsGermInjectiveAt x
      U : X.Opens
      hxU : Membership.mem U x
      φ : Quiver.Hom A (X.presheaf.stalk x)
      φRA : Quiver.Hom R A
      φRX : Quiver.Hom R (X.presheaf.obj { unop := U })
      hφRA : φRA.hom.FiniteType
      e✝ : Eq (CategoryTheory.CategoryStruct.comp φRA φ) (CategoryTheory.CategoryStr …
      V : X.Opens
      hxV : Membership.mem V x
      iVU : LE.le V U
      hV : LE.le φ.hom.range (X.presheaf.germ V x hxV).hom.range
      V' : X.Opens
      hxV' : Membership.mem V' x
      hV' : AlgebraicGeometry.IsAffineOpen V'
      iV'V : LE.le V' V
      H : Function.Injective ⇑(X.presheaf.germ V' x hxV').hom
      f : Quiver.Hom (X.presheaf.obj { unop := V' }) (X.presheaf.stalk x) := X.presh …
      hf' : LE.le (X.presheaf.germ V x hxV).hom.range f.hom.range
      e : RingEquiv (↑(X.presheaf.obj { unop := V' })) (Subtype fun x_1 => Membershi …
      a : ↑R
      ⊢ Eq (e ((CategoryTheory.CategoryStruct.comp φRX (X.presheaf.map ⋯.hom.op)).ho …
    -/
    ext
    /-
      case intro.intro.intro.intro.intro.intro.intro.refine_2.hf.a.a.a
      X : AlgebraicGeometry.Scheme
      R A : CommRingCat
      x : ↑↑X.toPresheafedSpace
      inst✝ : X.IsGermInjectiveAt x
      U : X.Opens
      hxU : Membership.mem U x
      φ : Quiver.Hom A (X.presheaf.stalk x)
      φRA : Quiver.Hom R A
      φRX : Quiver.Hom R (X.presheaf.obj { unop := U })
      hφRA : φRA.hom.FiniteType
      e✝ : Eq (CategoryTheory.CategoryStruct.comp φRA φ) (CategoryTheory.CategoryStr …
      V : X.Opens
      hxV : Membership.mem V x
      iVU : LE.le V U
      hV : LE.le φ.hom.range (X.presheaf.germ V x hxV).hom.range
      V' : X.Opens
      hxV' : Membership.mem V' x
      hV' : AlgebraicGeometry.IsAffineOpen V'
      iV'V : LE.le V' V
      H : Function.Injective ⇑(X.presheaf.germ V' x hxV').hom
      f : Quiver.Hom (X.presheaf.obj { unop := V' }) (X.presheaf.stalk x) := X.presh …
      hf' : LE.le (X.presheaf.germ V x hxV).hom.range f.hom.range
      e : RingEquiv (↑(X.presheaf.obj { unop := V' })) (Subtype fun x_1 => Membershi …
      a : ↑R
      ⊢ Eq ↑(e ((CategoryTheory.CategoryStruct.comp φRX (X.presheaf.map ⋯.hom.op)).h …
    -/
    show X.presheaf.germ _ _ _ (X.presheaf.map _ _) = (φRA ≫ φ) a
    /-
      case intro.intro.intro.intro.intro.intro.intro.refine_2.hf.a.a.a
      X : AlgebraicGeometry.Scheme
      R A : CommRingCat
      x : ↑↑X.toPresheafedSpace
      inst✝ : X.IsGermInjectiveAt x
      U : X.Opens
      hxU : Membership.mem U x
      φ : Quiver.Hom A (X.presheaf.stalk x)
      φRA : Quiver.Hom R A
      φRX : Quiver.Hom R (X.presheaf.obj { unop := U })
      hφRA : φRA.hom.FiniteType
      e✝ : Eq (CategoryTheory.CategoryStruct.comp φRA φ) (CategoryTheory.CategoryStr …
      V : X.Opens
      hxV : Membership.mem V x
      iVU : LE.le V U
      hV : LE.le φ.hom.range (X.presheaf.germ V x hxV).hom.range
      V' : X.Opens
      hxV' : Membership.mem V' x
      hV' : AlgebraicGeometry.IsAffineOpen V'
      iV'V : LE.le V' V
      H : Function.Injective ⇑(X.presheaf.germ V' x hxV').hom
      f : Quiver.Hom (X.presheaf.obj { unop := V' }) (X.presheaf.stalk x) := X.presh …
      hf' : LE.le (X.presheaf.germ V x hxV).hom.range f.hom.range
      e : RingEquiv (↑(X.presheaf.obj { unop := V' })) (Subtype fun x_1 => Membershi …
      a : ↑R
      ⊢ Eq ((X.presheaf.germ V' x hxV').hom ((X.presheaf.map ⋯.hom.op).hom (φRX.hom  …
    -/
    rw [CommRingCat.germ_res_apply, ‹φRA ≫ φ = _›]
    /-
      case intro.intro.intro.intro.intro.intro.intro.refine_2.hf.a.a.a
      X : AlgebraicGeometry.Scheme
      R A : CommRingCat
      x : ↑↑X.toPresheafedSpace
      inst✝ : X.IsGermInjectiveAt x
      U : X.Opens
      hxU : Membership.mem U x
      φ : Quiver.Hom A (X.presheaf.stalk x)
      φRA : Quiver.Hom R A
      φRX : Quiver.Hom R (X.presheaf.obj { unop := U })
      hφRA : φRA.hom.FiniteType
      e✝ : Eq (CategoryTheory.CategoryStruct.comp φRA φ) (CategoryTheory.CategoryStr …
      V : X.Opens
      hxV : Membership.mem V x
      iVU : LE.le V U
      hV : LE.le φ.hom.range (X.presheaf.germ V x hxV).hom.range
      V' : X.Opens
      hxV' : Membership.mem V' x
      hV' : AlgebraicGeometry.IsAffineOpen V'
      iV'V : LE.le V' V
      H : Function.Injective ⇑(X.presheaf.germ V' x hxV').hom
      f : Quiver.Hom (X.presheaf.obj { unop := V' }) (X.presheaf.stalk x) := X.presh …
      hf' : LE.le (X.presheaf.germ V x hxV).hom.range f.hom.range
      e : RingEquiv (↑(X.presheaf.obj { unop := V' })) (Subtype fun x_1 => Membershi …
      a : ↑R
      ⊢ Eq ((X.presheaf.germ U x ⋯).hom (φRX.hom a)) ((CategoryTheory.CategoryStruct …
    -/
    rfl
    /-
      🎉 no goals
    -/


/--
Given `S`-schemes `X Y` and points `x : X` `y : Y` over `s : S`.
Suppose we have the following diagram of `S`-schemes
```
Spec 𝒪_{X, x} ⟶ X
    |
  Spec(φ)
    ↓
Spec 𝒪_{Y, y} ⟶ Y
```
Then the map `Spec(φ)` spreads out to an `S`-morphism on an open subscheme `U ⊆ X`,
```
Spec 𝒪_{X, x} ⟶ U ⊆ X
    |             |
  Spec(φ)         |
    ↓             ↓
Spec 𝒪_{Y, y} ⟶ Y
```
provided that `Y` is locally of finite type over `S` and
`X` is "germ-injective" at `x` (e.g. when it's integral or locally noetherian).

TODO: The condition on `X` is unnecessary when `Y` is locally of finite presentation.
-/
@[stacks 0BX6]
lemma spread_out_of_isGermInjective [LocallyOfFiniteType sY] {x : X} [X.IsGermInjectiveAt x] {y : Y}
    (e : sX.base x = sY.base y) (φ : Y.presheaf.stalk y ⟶ X.presheaf.stalk x)
    (h : sY.stalkMap y ≫ φ =
      S.presheaf.stalkSpecializes (Inseparable.of_eq e).specializes ≫ sX.stalkMap x) :
    ∃ (U : X.Opens) (hxU : x ∈ U) (f : U.toScheme ⟶ Y),
      Spec.map φ ≫ Y.fromSpecStalk y = U.fromSpecStalkOfMem x hxU ≫ f ∧
      f ≫ sY = U.ι ≫ sX := by
  obtain ⟨_, ⟨U, hU, rfl⟩, hxU, -⟩ :=
    (isBasis_affine_open S).exists_subset_of_mem_open (Set.mem_univ (sX.base x)) isOpen_univ
  /-
    case intro.intro.intro.intro.intro
    X Y S : AlgebraicGeometry.Scheme
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    inst✝¹ : AlgebraicGeometry.LocallyOfFiniteType sY
    x : ↑↑X.toPresheafedSpace
    inst✝ : X.IsGermInjectiveAt x
    y : ↑↑Y.toPresheafedSpace
    e : Eq (sX.base x) (sY.base y)
    φ : Quiver.Hom (Y.presheaf.stalk y) (X.presheaf.stalk x)
    h : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.stalk …
    U : TopologicalSpace.Opens ↑↑S.toPresheafedSpace
    hU : Membership.mem S.affineOpens U
    hxU : Membership.mem (↑U) (sX.base x)
    ⊢ Exists fun U => Exists fun hxU => Exists fun f => And (Eq (CategoryTheory.Ca …
  -/
  have hyU : sY.base y ∈ U := e ▸ hxU
  obtain ⟨_, ⟨V : Y.Opens, hV, rfl⟩, hyV, iVU⟩ :=
    (isBasis_affine_open Y).exists_subset_of_mem_open hyU (sY ⁻¹ᵁ U).2
  have : sY.appLE U V iVU ≫ Y.presheaf.germ V y hyV ≫ φ =
      sX.app U ≫ X.presheaf.germ (sX ⁻¹ᵁ U) x hxU := by
    rw [Scheme.Hom.appLE, Category.assoc, Y.presheaf.germ_res_assoc,
      ← Scheme.stalkMap_germ_assoc, h]
    simp
  obtain ⟨W, hxW, φ', i, hW, h₁, h₂⟩ :=
    exists_lift_of_germInjective (R := Γ(S, U)) (A := Γ(Y, V)) (U := sX ⁻¹ᵁ U) (x := x) hxU
    (Y.presheaf.germ _ y hyV ≫ φ) (sY.appLE U V iVU) (sX.app U)
    (LocallyOfFiniteType.finiteType_of_affine_subset ⟨_, hU⟩ ⟨_, hV⟩ _) this
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    X Y S : AlgebraicGeometry.Scheme
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    inst✝¹ : AlgebraicGeometry.LocallyOfFiniteType sY
    x : ↑↑X.toPresheafedSpace
    inst✝ : X.IsGermInjectiveAt x
    y : ↑↑Y.toPresheafedSpace
    e : Eq (sX.base x) (sY.base y)
    φ : Quiver.Hom (Y.presheaf.stalk y) (X.presheaf.stalk x)
    h : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.stalk …
    U : TopologicalSpace.Opens ↑↑S.toPresheafedSpace
    hU : Membership.mem S.affineOpens U
    hxU : Membership.mem (↑U) (sX.base x)
    hyU : Membership.mem U (sY.base y)
    V : Y.Opens
    hV : Membership.mem Y.affineOpens V
    hyV : Membership.mem (↑V) y
    iVU : HasSubset.Subset (↑V) ((TopologicalSpace.Opens.map sY.base).obj U).carrier
    this : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.ap …
    W : X.Opens
    hxW : Membership.mem W x
    φ' : Quiver.Hom (Y.presheaf.obj { unop := V }) (X.presheaf.obj { unop := W })
    i : LE.le W ((TopologicalSpace.Opens.map sX.base).obj U)
    hW : AlgebraicGeometry.IsAffineOpen W
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (Y.presheaf.germ V y hyV) φ) (Cate …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.app  …
    ⊢ Exists fun U => Exists fun hxU => Exists fun f => And (Eq (CategoryTheory.Ca …
  -/
  refine ⟨W, hxW, W.toSpecΓ ≫ Spec.map φ' ≫ hV.fromSpec, ?_, ?_⟩
  · rw [W.fromSpecStalkOfMem_toSpecΓ_assoc x hxW, ← Spec.map_comp_assoc, ← h₁,
      Spec.map_comp, Category.assoc, ← IsAffineOpen.fromSpecStalk,
      IsAffineOpen.fromSpecStalk_eq_fromSpecStalk]
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      X Y S : AlgebraicGeometry.Scheme
      sX : Quiver.Hom X S
      sY : Quiver.Hom Y S
      inst✝¹ : AlgebraicGeometry.LocallyOfFiniteType sY
      x : ↑↑X.toPresheafedSpace
      inst✝ : X.IsGermInjectiveAt x
      y : ↑↑Y.toPresheafedSpace
      e : Eq (sX.base x) (sY.base y)
      φ : Quiver.Hom (Y.presheaf.stalk y) (X.presheaf.stalk x)
      h : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.stalk …
      U : TopologicalSpace.Opens ↑↑S.toPresheafedSpace
      hU : Membership.mem S.affineOpens U
      hxU : Membership.mem (↑U) (sX.base x)
      hyU : Membership.mem U (sY.base y)
      V : Y.Opens
      hV : Membership.mem Y.affineOpens V
      hyV : Membership.mem (↑V) y
      iVU : HasSubset.Subset (↑V) ((TopologicalSpace.Opens.map sY.base).obj U).carrier
      this : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.ap …
      W : X.Opens
      hxW : Membership.mem W x
      φ' : Quiver.Hom (Y.presheaf.obj { unop := V }) (X.presheaf.obj { unop := W })
      i : LE.le W ((TopologicalSpace.Opens.map sX.base).obj U)
      hW : AlgebraicGeometry.IsAffineOpen W
      h₁ : Eq (CategoryTheory.CategoryStruct.comp (Y.presheaf.germ V y hyV) φ) (Cate …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.app  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp W …
    -/
  · simp only [Category.assoc, IsAffineOpen.isoSpec_inv_ι_assoc]
    rw [← IsAffineOpen.Spec_map_appLE_fromSpec sY hU hV iVU, ← Spec.map_comp_assoc, ← h₂,
      ← Scheme.Hom.appLE, ← hW.isoSpec_hom, IsAffineOpen.Spec_map_appLE_fromSpec sX hU hW i,
      ← Iso.eq_inv_comp, IsAffineOpen.isoSpec_inv_ι_assoc]


/--
Given `S`-schemes `X Y`, a point `x : X`, and a `S`-morphism `φ : Spec 𝒪_{X, x} ⟶ Y`,
we may spread it out to an `S`-morphism `f : U ⟶ Y`
provided that `Y` is locally of finite type over `S` and
`X` is "germ-injective" at `x` (e.g. when it's integral or locally noetherian).

TODO: The condition on `X` is unnecessary when `Y` is locally of finite presentation.
-/
lemma spread_out_of_isGermInjective' [LocallyOfFiniteType sY] {x : X} [X.IsGermInjectiveAt x]
    (φ : Spec (X.presheaf.stalk x) ⟶ Y)
    (h : φ ≫ sY = X.fromSpecStalk x ≫ sX) :
    ∃ (U : X.Opens) (hxU : x ∈ U) (f : U.toScheme ⟶ Y),
      φ = U.fromSpecStalkOfMem x hxU ≫ f ∧ f ≫ sY = U.ι ≫ sX := by
  /-
    X Y S : AlgebraicGeometry.Scheme
    sX : Quiver.Hom X S
    sY : Quiver.Hom Y S
    inst✝¹ : AlgebraicGeometry.LocallyOfFiniteType sY
    x : ↑↑X.toPresheafedSpace
    inst✝ : X.IsGermInjectiveAt x
    φ : Quiver.Hom (AlgebraicGeometry.Spec (X.presheaf.stalk x)) Y
    h : Eq (CategoryTheory.CategoryStruct.comp φ sY) (CategoryTheory.CategoryStruc …
    ⊢ Exists fun U => Exists fun hxU => Exists fun f => And (Eq φ (CategoryTheory. …
  -/
  have := spread_out_of_isGermInjective sX sY ?_ (Scheme.stalkClosedPointTo φ) ?_
    /-
      case refine_3
      X Y S : AlgebraicGeometry.Scheme
      sX : Quiver.Hom X S
      sY : Quiver.Hom Y S
      inst✝¹ : AlgebraicGeometry.LocallyOfFiniteType sY
      x : ↑↑X.toPresheafedSpace
      inst✝ : X.IsGermInjectiveAt x
      φ : Quiver.Hom (AlgebraicGeometry.Spec (X.presheaf.stalk x)) Y
      h : Eq (CategoryTheory.CategoryStruct.comp φ sY) (CategoryTheory.CategoryStruc …
      this : Exists fun U => Exists fun hxU => Exists fun f => And (Eq (CategoryTheo …
      ⊢ Exists fun U => Exists fun hxU => Exists fun f => And (Eq φ (CategoryTheory. …
    -/
  · simpa only [Scheme.Spec_stalkClosedPointTo_fromSpecStalk] using this
    /-
      🎉 no goals
    -/
    /-
      case refine_1
      X Y S : AlgebraicGeometry.Scheme
      sX : Quiver.Hom X S
      sY : Quiver.Hom Y S
      inst✝¹ : AlgebraicGeometry.LocallyOfFiniteType sY
      x : ↑↑X.toPresheafedSpace
      inst✝ : X.IsGermInjectiveAt x
      φ : Quiver.Hom (AlgebraicGeometry.Spec (X.presheaf.stalk x)) Y
      h : Eq (CategoryTheory.CategoryStruct.comp φ sY) (CategoryTheory.CategoryStruc …
      ⊢ Eq (sX.base x) (sY.base (φ.base (IsLocalRing.closedPoint ↑(X.presheaf.stalk  …
    -/
  · rw [← Scheme.comp_base_apply, h, Scheme.comp_base_apply, Scheme.fromSpecStalk_closedPoint]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X Y S : AlgebraicGeometry.Scheme
      sX : Quiver.Hom X S
      sY : Quiver.Hom Y S
      inst✝¹ : AlgebraicGeometry.LocallyOfFiniteType sY
      x : ↑↑X.toPresheafedSpace
      inst✝ : X.IsGermInjectiveAt x
      φ : Quiver.Hom (AlgebraicGeometry.Spec (X.presheaf.stalk x)) Y
      h : Eq (CategoryTheory.CategoryStruct.comp φ sY) (CategoryTheory.CategoryStruc …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.stalkMa …
    -/
  · apply Spec.map_injective
    /-
      case refine_2.a
      X Y S : AlgebraicGeometry.Scheme
      sX : Quiver.Hom X S
      sY : Quiver.Hom Y S
      inst✝¹ : AlgebraicGeometry.LocallyOfFiniteType sY
      x : ↑↑X.toPresheafedSpace
      inst✝ : X.IsGermInjectiveAt x
      φ : Quiver.Hom (AlgebraicGeometry.Spec (X.presheaf.stalk x)) Y
      h : Eq (CategoryTheory.CategoryStruct.comp φ sY) (CategoryTheory.CategoryStruc …
      ⊢ Eq (AlgebraicGeometry.Spec.map (CategoryTheory.CategoryStruct.comp (Algebrai …
    -/
    rw [← cancel_mono (S.fromSpecStalk _)]
    simpa only [Spec.map_comp, Category.assoc, Scheme.Spec_map_stalkMap_fromSpecStalk,
      Scheme.Spec_stalkClosedPointTo_fromSpecStalk_assoc,
      Scheme.Spec_map_stalkSpecializes_fromSpecStalk]


