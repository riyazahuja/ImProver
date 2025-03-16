/-- Definition of a (Pre)Galois category. Lenstra, Def 3.1, (G1)-(G3) -/
class PreGaloisCategory (C : Type u₁) [Category.{u₂, u₁} C] : Prop where
  /-- `C` has a terminal object (G1). -/
  hasTerminal : HasTerminal C := by infer_instance
  /-- `C` has pullbacks (G1). -/
  hasPullbacks : HasPullbacks C := by infer_instance
  /-- `C` has finite coproducts (G2). -/
  hasFiniteCoproducts : HasFiniteCoproducts C := by infer_instance
  /-- `C` has quotients by finite groups (G2). -/
  hasQuotientsByFiniteGroups (G : Type u₂) [Group G] [Finite G] :
    HasColimitsOfShape (SingleObj G) C := by infer_instance
  /-- Every monomorphism in `C` induces an isomorphism on a direct summand (G3). -/
  monoInducesIsoOnDirectSummand {X Y : C} (i : X ⟶ Y) [Mono i] : ∃ (Z : C) (u : Z ⟶ Y),
    Nonempty (IsColimit (BinaryCofan.mk i u))


/-- Definition of a fiber functor from a Galois category. Lenstra, Def 3.1, (G4)-(G6) -/
class FiberFunctor {C : Type u₁} [Category.{u₂, u₁} C] [PreGaloisCategory C]
    (F : C ⥤ FintypeCat.{w}) where
  /-- `F` preserves terminal objects (G4). -/
  preservesTerminalObjects : PreservesLimitsOfShape (CategoryTheory.Discrete PEmpty.{1}) F := by
    infer_instance
  /-- `F` preserves pullbacks (G4). -/
  preservesPullbacks : PreservesLimitsOfShape WalkingCospan F := by infer_instance
  /-- `F` preserves finite coproducts (G5). -/
  preservesFiniteCoproducts : PreservesFiniteCoproducts F := by infer_instance
  /-- `F` preserves epimorphisms (G5). -/
  preservesEpis : Functor.PreservesEpimorphisms F := by infer_instance
  /-- `F` preserves quotients by finite groups (G5). -/
  preservesQuotientsByFiniteGroups (G : Type u₂) [Group G] [Finite G] :
    PreservesColimitsOfShape (SingleObj G) F := by infer_instance
  /-- `F` reflects isomorphisms (G6). -/
  reflectsIsos : F.ReflectsIsomorphisms := by infer_instance


/-- An object of a category `C` is connected if it is not initial
and has no non-trivial subobjects. Lenstra, 3.12. -/
class IsConnected {C : Type u₁} [Category.{u₂, u₁} C] (X : C) : Prop where
  /-- `X` is not an initial object. -/
  notInitial : IsInitial X → False
  /-- `X` has no non-trivial subobjects. -/
  noTrivialComponent (Y : C) (i : Y ⟶ X) [Mono i] : (IsInitial Y → False) → IsIso i


/-- A functor is said to preserve connectedness if whenever `X : C` is connected,
also `F.obj X` is connected. -/
class PreservesIsConnected {C : Type u₁} [Category.{u₂, u₁} C] {D : Type v₁}
    [Category.{v₂, v₁} D] (F : C ⥤ D) : Prop where
  /-- `F.obj X` is connected if `X` is connected. -/
  preserves : ∀ {X : C} [IsConnected X], IsConnected (F.obj X)


instance : HasFiniteLimits C := hasFiniteLimits_of_hasTerminal_and_pullbacks


instance : HasBinaryProducts C := hasBinaryProducts_of_hasTerminal_and_pullbacks C


instance : HasEqualizers C := hasEqualizers_of_hasPullbacks_and_binary_products

-- A `PreGaloisCategory` has quotients by finite groups in arbitrary universes. -/

instance {G : Type*} [Group G] [Finite G] : HasColimitsOfShape (SingleObj G) C := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.PreGaloisCategory C
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite G
    ⊢ CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.SingleObj G) C
  -/
  obtain ⟨G', hg, hf, ⟨e⟩⟩ := Finite.exists_type_univ_nonempty_mulEquiv G
  /-
    case intro.intro.intro.intro
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.PreGaloisCategory C
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite G
    G' : Type ?u.5493
    hg : Group G'
    hf : Fintype G'
    e : MulEquiv G G'
    ⊢ CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.SingleObj G) C
  -/
  exact Limits.hasColimitsOfShape_of_equivalence e.toSingleObjEquiv.symm
  /-
    🎉 no goals
  -/


noncomputable instance : ReflectsLimitsOfShape (Discrete PEmpty.{1}) F :=
  reflectsLimitsOfShape_of_reflectsIsomorphisms


noncomputable instance : ReflectsColimitsOfShape (Discrete PEmpty.{1}) F :=
  reflectsColimitsOfShape_of_reflectsIsomorphisms


noncomputable instance : PreservesFiniteLimits F :=
  preservesFiniteLimits_of_preservesTerminal_and_pullbacks F


/-- Fiber functors preserve quotients by finite groups in arbitrary universes. -/
instance {G : Type*} [Group G] [Finite G] :
    PreservesColimitsOfShape (SingleObj G) F := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝³ : CategoryTheory.PreGaloisCategory C
    inst✝² : CategoryTheory.PreGaloisCategory.FiberFunctor F
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite G
    ⊢ CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.SingleObj G) F
  -/
  choose G' hg hf he using Finite.exists_type_univ_nonempty_mulEquiv G
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝³ : CategoryTheory.PreGaloisCategory C
    inst✝² : CategoryTheory.PreGaloisCategory.FiberFunctor F
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite G
    G' : Type ?u.11769
    hg : Group G'
    hf : Fintype G'
    he : Nonempty (MulEquiv G G')
    ⊢ CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.SingleObj G) F
  -/
  exact Limits.preservesColimitsOfShape_of_equiv he.some.toSingleObjEquiv.symm F
  /-
    🎉 no goals
  -/


/-- Fiber functors reflect monomorphisms. -/
instance : ReflectsMonomorphisms F := ReflectsMonomorphisms.mk <| by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Mono (F.map f) → CategoryTh …
  -/
  intro X Y f _
  haveI : IsIso (pullback.fst (F.map f) (F.map f)) :=
    isIso_fst_of_mono (F.map f)
  haveI : IsIso (F.map (pullback.fst f f)) := by
    rw [← PreservesPullback.iso_hom_fst]
    exact IsIso.comp_isIso
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X Y : C
    f : Quiver.Hom X Y
    a✝ : CategoryTheory.Mono (F.map f)
    this✝ : CategoryTheory.IsIso (CategoryTheory.Limits.pullback.fst (F.map f) (F. …
    this : CategoryTheory.IsIso (F.map (CategoryTheory.Limits.pullback.fst f f))
    ⊢ CategoryTheory.Mono f
  -/
  haveI : IsIso (pullback.fst f f) := isIso_of_reflects_iso (pullback.fst _ _) F
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X Y : C
    f : Quiver.Hom X Y
    a✝ : CategoryTheory.Mono (F.map f)
    this✝¹ : CategoryTheory.IsIso (CategoryTheory.Limits.pullback.fst (F.map f) (F …
    this✝ : CategoryTheory.IsIso (F.map (CategoryTheory.Limits.pullback.fst f f))
    this : CategoryTheory.IsIso (CategoryTheory.Limits.pullback.fst f f)
    ⊢ CategoryTheory.Mono f
  -/
  exact (pullback.diagonal_isKernelPair f).mono_of_isIso_fst
  /-
    🎉 no goals
  -/


/-- Fiber functors are faithful. -/
instance : F.Faithful where
  map_injective {X Y} f g h := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.PreGaloisCategory C
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X Y : C
      f g : Quiver.Hom X Y
      h : Eq (F.map f) (F.map g)
      ⊢ Eq f g
    -/
    haveI : IsIso (equalizer.ι (F.map f) (F.map g)) := equalizer.ι_of_eq h
    haveI : IsIso (F.map (equalizer.ι f g)) := by
      rw [← equalizerComparison_comp_π f g F]
      exact IsIso.comp_isIso
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.PreGaloisCategory C
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X Y : C
      f g : Quiver.Hom X Y
      h : Eq (F.map f) (F.map g)
      this✝ : CategoryTheory.IsIso (CategoryTheory.Limits.equalizer.ι (F.map f) (F.m …
      this : CategoryTheory.IsIso (F.map (CategoryTheory.Limits.equalizer.ι f g))
      ⊢ Eq f g
    -/
    haveI : IsIso (equalizer.ι f g) := isIso_of_reflects_iso _ F
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.PreGaloisCategory C
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X Y : C
      f g : Quiver.Hom X Y
      h : Eq (F.map f) (F.map g)
      this✝¹ : CategoryTheory.IsIso (CategoryTheory.Limits.equalizer.ι (F.map f) (F. …
      this✝ : CategoryTheory.IsIso (F.map (CategoryTheory.Limits.equalizer.ι f g))
      this : CategoryTheory.IsIso (CategoryTheory.Limits.equalizer.ι f g)
      ⊢ Eq f g
    -/
    exact eq_of_epi_equalizer
    /-
      🎉 no goals
    -/


/-- If `F` is a fiber functor and `E` is an equivalence between categories of finite types,
then `F ⋙ E` is again a fiber functor. -/
lemma comp_right (E : FintypeCat.{w} ⥤ FintypeCat.{t}) [E.IsEquivalence] :
    FiberFunctor (F ⋙ E) where
  preservesQuotientsByFiniteGroups _ := comp_preservesColimitsOfShape F E


/-- The canonical action of `Aut F` on the fiber of each object. -/
instance (X : C) : MulAction (Aut F) (F.obj X) where
  smul σ x := σ.hom.app X x
  one_smul _ := rfl
  mul_smul _ _ _ := rfl


lemma mulAction_def {X : C} (σ : Aut F) (x : F.obj X) :
    σ • x = σ.hom.app X x :=
  rfl


lemma mulAction_naturality {X Y : C} (σ : Aut F) (f : X ⟶ Y) (x : F.obj X) :
    σ • F.map f x = F.map f (σ • x) :=
  FunctorToFintypeCat.naturality F F σ.hom f x


/-- An object that is neither initial or connected has a non-trivial subobject. -/
lemma has_non_trivial_subobject_of_not_isConnected_of_not_initial (X : C) (hc : ¬ IsConnected X)
    (hi : IsInitial X → False) :
    ∃ (Y : C) (v : Y ⟶ X), (IsInitial Y → False) ∧ Mono v ∧ (¬ IsIso v) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{u₂, u₁} C
    X : C
    hc : Not (CategoryTheory.PreGaloisCategory.IsConnected X)
    hi : CategoryTheory.Limits.IsInitial X → False
    ⊢ Exists fun Y => Exists fun v => And (CategoryTheory.Limits.IsInitial Y → Fal …
  -/
  contrapose! hc
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{u₂, u₁} C
    X : C
    hi : CategoryTheory.Limits.IsInitial X → False
    hc : ∀ (Y : C) (v : Quiver.Hom Y X), (CategoryTheory.Limits.IsInitial Y → Fals …
    ⊢ CategoryTheory.PreGaloisCategory.IsConnected X
  -/
  exact ⟨hi, fun Y i hm hni ↦ hc Y i hni hm⟩
  /-
    🎉 no goals
  -/


/-- The cardinality of the fiber is preserved under isomorphisms. -/
lemma card_fiber_eq_of_iso {X Y : C} (i : X ≅ Y) : Nat.card (F.obj X) = Nat.card (F.obj Y) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    X Y : C
    i : CategoryTheory.Iso X Y
    ⊢ Eq (Nat.card ↑(F.obj X)) (Nat.card ↑(F.obj Y))
  -/
  have e : F.obj X ≃ F.obj Y := Iso.toEquiv (mapIso (F ⋙ FintypeCat.incl) i)
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    X Y : C
    i : CategoryTheory.Iso X Y
    e : Equiv ↑(F.obj X) ↑(F.obj Y)
    ⊢ Eq (Nat.card ↑(F.obj X)) (Nat.card ↑(F.obj Y))
  -/
  exact Nat.card_eq_of_bijective e (Equiv.bijective e)
  /-
    🎉 no goals
  -/


/-- An object is initial if and only if its fiber is empty. -/
lemma initial_iff_fiber_empty (X : C) : Nonempty (IsInitial X) ↔ IsEmpty (F.obj X) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    ⊢ Iff (Nonempty (CategoryTheory.Limits.IsInitial X)) (IsEmpty ↑(F.obj X))
  -/
  rw [(IsInitial.isInitialIffObj F X).nonempty_congr]
  haveI : PreservesFiniteColimits (forget FintypeCat) := by
    show PreservesFiniteColimits FintypeCat.incl
    infer_instance
  haveI : ReflectsColimit (Functor.empty.{0} _) (forget FintypeCat) := by
    show ReflectsColimit (Functor.empty.{0} _) FintypeCat.incl
    infer_instance
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    this✝ : CategoryTheory.Limits.PreservesFiniteColimits (CategoryTheory.forget F …
    this : CategoryTheory.Limits.ReflectsColimit (CategoryTheory.Functor.empty Fin …
    ⊢ Iff (Nonempty (CategoryTheory.Limits.IsInitial (F.obj X))) (IsEmpty ↑(F.obj  …
  -/
  exact Concrete.initial_iff_empty_of_preserves_of_reflects (F.obj X)
  /-
    🎉 no goals
  -/


/-- An object is not initial if and only if its fiber is nonempty. -/
lemma not_initial_iff_fiber_nonempty (X : C) : (IsInitial X → False) ↔ Nonempty (F.obj X) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    ⊢ Iff (CategoryTheory.Limits.IsInitial X → False) (Nonempty ↑(F.obj X))
  -/
  rw [← not_isEmpty_iff]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    ⊢ Iff (CategoryTheory.Limits.IsInitial X → False) (Not (IsEmpty ↑(F.obj X)))
  -/
  refine ⟨fun h he ↦ ?_, fun h hin ↦ h <| (initial_iff_fiber_empty F X).mp ⟨hin⟩⟩
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    h : CategoryTheory.Limits.IsInitial X → False
    he : IsEmpty ↑(F.obj X)
    ⊢ False
  -/
  exact Nonempty.elim ((initial_iff_fiber_empty F X).mpr he) h
  /-
    🎉 no goals
  -/


/-- An object whose fiber is inhabited is not initial. -/
lemma not_initial_of_inhabited {X : C} (x : F.obj X) (h : IsInitial X) : False :=
  ((initial_iff_fiber_empty F X).mp ⟨h⟩).false x


/-- The fiber of a connected object is nonempty. -/
instance nonempty_fiber_of_isConnected (X : C) [IsConnected X] : Nonempty (F.obj X) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝² : CategoryTheory.PreGaloisCategory C
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
    ⊢ Nonempty ↑(F.obj X)
  -/
  by_contra h
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝² : CategoryTheory.PreGaloisCategory C
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
    h : Not (Nonempty ↑(F.obj X))
    ⊢ False
  -/
  have ⟨hin⟩ : Nonempty (IsInitial X) := (initial_iff_fiber_empty F X).mpr (not_nonempty_iff.mp h)
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝² : CategoryTheory.PreGaloisCategory C
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
    h : Not (Nonempty ↑(F.obj X))
    hin : CategoryTheory.Limits.IsInitial X
    ⊢ False
  -/
  exact IsConnected.notInitial hin
  /-
    🎉 no goals
  -/


/-- The fiber of the equalizer of `f g : X ⟶ Y` is equivalent to the set of agreement of `f`
and `g`. -/
noncomputable def fiberEqualizerEquiv {X Y : C} (f g : X ⟶ Y) :
    F.obj (equalizer f g) ≃ { x : F.obj X // F.map f x = F.map g x } :=
  (PreservesEqualizer.iso (F ⋙ FintypeCat.incl) f g ≪≫
  Types.equalizerIso (F.map f) (F.map g)).toEquiv


@[simp]
lemma fiberEqualizerEquiv_symm_ι_apply {X Y : C} {f g : X ⟶ Y} (x : F.obj X)
    (h : F.map f x = F.map g x) :
    F.map (equalizer.ι f g) ((fiberEqualizerEquiv F f g).symm ⟨x, h⟩) = x := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X Y : C
    f g : Quiver.Hom X Y
    x : ↑(F.obj X)
    h : Eq (F.map f x) (F.map g x)
    ⊢ Eq (F.map (CategoryTheory.Limits.equalizer.ι f g) ((CategoryTheory.PreGalois …
  -/
  simp [fiberEqualizerEquiv]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X Y : C
    f g : Quiver.Hom X Y
    x : ↑(F.obj X)
    h : Eq (F.map f x) (F.map g x)
    ⊢ Eq (F.map (CategoryTheory.Limits.equalizer.ι f g) ((CategoryTheory.Limits.Pr …
  -/
  change ((Types.equalizerIso _ _).inv ≫ _ ≫ (F ⋙ FintypeCat.incl).map (equalizer.ι f g)) _ = _
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X Y : C
    f g : Quiver.Hom X Y
    x : ↑(F.obj X)
    h : Eq (F.map f x) (F.map g x)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Types.equalize …
  -/
  erw [PreservesEqualizer.iso_inv_ι, Types.equalizerIso_inv_comp_ι]
  /-
    🎉 no goals
  -/


/-- The fiber of the pullback is the fiber product of the fibers. -/
noncomputable def fiberPullbackEquiv {X A B : C} (f : A ⟶ X) (g : B ⟶ X) :
    F.obj (pullback f g) ≃ { p : F.obj A × F.obj B // F.map f p.1 = F.map g p.2 } :=
  (PreservesPullback.iso (F ⋙ FintypeCat.incl) f g ≪≫
  Types.pullbackIsoPullback (F.map f) (F.map g)).toEquiv


@[simp]
lemma fiberPullbackEquiv_symm_fst_apply {X A B : C} {f : A ⟶ X} {g : B ⟶ X}
    (a : F.obj A) (b : F.obj B) (h : F.map f a = F.map g b) :
    F.map (pullback.fst f g) ((fiberPullbackEquiv F f g).symm ⟨(a, b), h⟩) = a := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A B : C
    f : Quiver.Hom A X
    g : Quiver.Hom B X
    a : ↑(F.obj A)
    b : ↑(F.obj B)
    h : Eq (F.map f a) (F.map g b)
    ⊢ Eq (F.map (CategoryTheory.Limits.pullback.fst f g) ((CategoryTheory.PreGaloi …
  -/
  simp [fiberPullbackEquiv]
  change ((Types.pullbackIsoPullback _ _).inv ≫ _ ≫
    (F ⋙ FintypeCat.incl).map (pullback.fst f g)) _ = _
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A B : C
    f : Quiver.Hom A X
    g : Quiver.Hom B X
    a : ↑(F.obj A)
    b : ↑(F.obj B)
    h : Eq (F.map f a) (F.map g b)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Types.pullback …
  -/
  erw [PreservesPullback.iso_inv_fst, Types.pullbackIsoPullback_inv_fst]
  /-
    🎉 no goals
  -/


@[simp]
lemma fiberPullbackEquiv_symm_snd_apply {X A B : C} {f : A ⟶ X} {g : B ⟶ X}
    (a : F.obj A) (b : F.obj B) (h : F.map f a = F.map g b) :
    F.map (pullback.snd f g) ((fiberPullbackEquiv F f g).symm ⟨(a, b), h⟩) = b := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A B : C
    f : Quiver.Hom A X
    g : Quiver.Hom B X
    a : ↑(F.obj A)
    b : ↑(F.obj B)
    h : Eq (F.map f a) (F.map g b)
    ⊢ Eq (F.map (CategoryTheory.Limits.pullback.snd f g) ((CategoryTheory.PreGaloi …
  -/
  simp [fiberPullbackEquiv]
  change ((Types.pullbackIsoPullback _ _).inv ≫ _ ≫
    (F ⋙ FintypeCat.incl).map (pullback.snd f g)) _ = _
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A B : C
    f : Quiver.Hom A X
    g : Quiver.Hom B X
    a : ↑(F.obj A)
    b : ↑(F.obj B)
    h : Eq (F.map f a) (F.map g b)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Types.pullback …
  -/
  erw [PreservesPullback.iso_inv_snd, Types.pullbackIsoPullback_inv_snd]
  /-
    🎉 no goals
  -/


/-- The fiber of the binary product is the binary product of the fibers. -/
noncomputable def fiberBinaryProductEquiv (X Y : C) :
    F.obj (X ⨯ Y) ≃ F.obj X × F.obj Y :=
  (PreservesLimitPair.iso (F ⋙ FintypeCat.incl) X Y ≪≫
  Types.binaryProductIso (F.obj X) (F.obj Y)).toEquiv


@[simp]
lemma fiberBinaryProductEquiv_symm_fst_apply {X Y : C} (x : F.obj X) (y : F.obj Y) :
    F.map prod.fst ((fiberBinaryProductEquiv F X Y).symm (x, y)) = x := by
  simp only [fiberBinaryProductEquiv, comp_obj, FintypeCat.incl_obj, Iso.toEquiv_comp,
    Equiv.symm_trans_apply, Iso.toEquiv_symm_fun]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X Y : C
    x : ↑(F.obj X)
    y : ↑(F.obj Y)
    ⊢ Eq (F.map CategoryTheory.Limits.prod.fst ((CategoryTheory.Limits.PreservesLi …
  -/
  change ((Types.binaryProductIso _ _).inv ≫ _ ≫ (F ⋙ FintypeCat.incl).map prod.fst) _ = _
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X Y : C
    x : ↑(F.obj X)
    y : ↑(F.obj Y)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Types.binaryPr …
  -/
  erw [PreservesLimitPair.iso_inv_fst, Types.binaryProductIso_inv_comp_fst]
  /-
    🎉 no goals
  -/


@[simp]
lemma fiberBinaryProductEquiv_symm_snd_apply {X Y : C} (x : F.obj X) (y : F.obj Y) :
    F.map prod.snd ((fiberBinaryProductEquiv F X Y).symm (x, y)) = y := by
  simp only [fiberBinaryProductEquiv, comp_obj, FintypeCat.incl_obj, Iso.toEquiv_comp,
    Equiv.symm_trans_apply, Iso.toEquiv_symm_fun]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X Y : C
    x : ↑(F.obj X)
    y : ↑(F.obj Y)
    ⊢ Eq (F.map CategoryTheory.Limits.prod.snd ((CategoryTheory.Limits.PreservesLi …
  -/
  change ((Types.binaryProductIso _ _).inv ≫ _ ≫ (F ⋙ FintypeCat.incl).map prod.snd) _ = _
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X Y : C
    x : ↑(F.obj X)
    y : ↑(F.obj Y)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Types.binaryPr …
  -/
  erw [PreservesLimitPair.iso_inv_snd, Types.binaryProductIso_inv_comp_snd]
  /-
    🎉 no goals
  -/


/-- The evaluation map is injective for connected objects. -/
lemma evaluation_injective_of_isConnected (A X : C) [IsConnected A] (a : F.obj A) :
    Function.Injective (fun (f : A ⟶ X) ↦ F.map f a) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝² : CategoryTheory.PreGaloisCategory C
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    A X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    a : ↑(F.obj A)
    ⊢ Function.Injective fun f => F.map f a
  -/
  intro f g (h : F.map f a = F.map g a)
  haveI : IsIso (equalizer.ι f g) := by
    apply IsConnected.noTrivialComponent _ (equalizer.ι f g)
    exact not_initial_of_inhabited F ((fiberEqualizerEquiv F f g).symm ⟨a, h⟩)
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝² : CategoryTheory.PreGaloisCategory C
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    A X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    a : ↑(F.obj A)
    f g : Quiver.Hom A X
    h : Eq (F.map f a) (F.map g a)
    this : CategoryTheory.IsIso (CategoryTheory.Limits.equalizer.ι f g)
    ⊢ Eq f g
  -/
  exact eq_of_epi_equalizer
  /-
    🎉 no goals
  -/


/-- The evaluation map on automorphisms is injective for connected objects. -/
lemma evaluation_aut_injective_of_isConnected (A : C) [IsConnected A] (a : F.obj A) :
    Function.Injective (fun f : Aut A ↦ F.map (f.hom) a) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝² : CategoryTheory.PreGaloisCategory C
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    A : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    a : ↑(F.obj A)
    ⊢ Function.Injective fun f => F.map f.hom a
  -/
  show Function.Injective ((fun f : A ⟶ A ↦ F.map f a) ∘ (fun f : Aut A ↦ f.hom))
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝² : CategoryTheory.PreGaloisCategory C
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    A : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    a : ↑(F.obj A)
    ⊢ Function.Injective (Function.comp (fun f => F.map f a) fun f => f.hom)
  -/
  apply Function.Injective.comp
    /-
      case hg
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝² : CategoryTheory.PreGaloisCategory C
      inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      A : C
      inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
      a : ↑(F.obj A)
      ⊢ Function.Injective fun f => F.map f a
    -/
  · exact evaluation_injective_of_isConnected F A A a
    /-
      🎉 no goals
    -/
    /-
      case hf
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝² : CategoryTheory.PreGaloisCategory C
      inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      A : C
      inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
      a : ↑(F.obj A)
      ⊢ Function.Injective fun f => f.hom
    -/
  · exact @Aut.ext _ _ A
    /-
      🎉 no goals
    -/


/-- A morphism from an object `X` with non-empty fiber to a connected object `A` is an
epimorphism. -/
lemma epi_of_nonempty_of_isConnected {X A : C} [IsConnected A] [h : Nonempty (F.obj X)]
    (f : X ⟶ A) : Epi f := Epi.mk <| fun {Z} u v huv ↦ by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝² : CategoryTheory.PreGaloisCategory C
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    h : Nonempty ↑(F.obj X)
    f : Quiver.Hom X A
    Z : C
    u v : Quiver.Hom A Z
    huv : Eq (CategoryTheory.CategoryStruct.comp f u) (CategoryTheory.CategoryStru …
    ⊢ Eq u v
  -/
  apply evaluation_injective_of_isConnected F A Z (F.map f (Classical.arbitrary _))
  /-
    case a
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝² : CategoryTheory.PreGaloisCategory C
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    h : Nonempty ↑(F.obj X)
    f : Quiver.Hom X A
    Z : C
    u v : Quiver.Hom A Z
    huv : Eq (CategoryTheory.CategoryStruct.comp f u) (CategoryTheory.CategoryStru …
    ⊢ Eq ((fun f_1 => F.map f_1 (F.map f (Classical.arbitrary ↑(F.obj X)))) u) ((f …
  -/
  simpa using congr_fun (F.congr_map huv) _
  /-
    🎉 no goals
  -/


/-- An epimorphism induces a surjective map on fibers. -/
lemma surjective_on_fiber_of_epi {X Y : C} (f : X ⟶ Y) [Epi f] : Function.Surjective (F.map f) :=
  surjective_of_epi (FintypeCat.incl.map (F.map f))

/- A morphism from an object with non-empty fiber to a connected object is surjective on fibers. -/

lemma surjective_of_nonempty_fiber_of_isConnected {X A : C} [Nonempty (F.obj X)]
    [IsConnected A] (f : X ⟶ A) :
    Function.Surjective (F.map f) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝³ : CategoryTheory.PreGaloisCategory C
    inst✝² : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A : C
    inst✝¹ : Nonempty ↑(F.obj X)
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    f : Quiver.Hom X A
    ⊢ Function.Surjective (F.map f)
  -/
  have : Epi f := epi_of_nonempty_of_isConnected F f
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝³ : CategoryTheory.PreGaloisCategory C
    inst✝² : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X A : C
    inst✝¹ : Nonempty ↑(F.obj X)
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    f : Quiver.Hom X A
    this : CategoryTheory.Epi f
    ⊢ Function.Surjective (F.map f)
  -/
  exact surjective_on_fiber_of_epi F f
  /-
    🎉 no goals
  -/


/-- If `X : ι → C` is a finite family of objects with non-empty fiber, then
also `∏ᶜ X` has non-empty fiber. -/
instance nonempty_fiber_pi_of_nonempty_of_finite {ι : Type*} [Finite ι] (X : ι → C)
    [∀ i, Nonempty (F.obj (X i))] : Nonempty (F.obj (∏ᶜ X)) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝³ : CategoryTheory.PreGaloisCategory C
    inst✝² : CategoryTheory.PreGaloisCategory.FiberFunctor F
    ι : Type u_1
    inst✝¹ : Finite ι
    X : ι → C
    inst✝ : ∀ (i : ι), Nonempty ↑(F.obj (X i))
    ⊢ Nonempty ↑(F.obj (CategoryTheory.Limits.piObj X))
  -/
  cases nonempty_fintype ι
  /-
    case intro
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝³ : CategoryTheory.PreGaloisCategory C
    inst✝² : CategoryTheory.PreGaloisCategory.FiberFunctor F
    ι : Type u_1
    inst✝¹ : Finite ι
    X : ι → C
    inst✝ : ∀ (i : ι), Nonempty ↑(F.obj (X i))
    val✝ : Fintype ι
    ⊢ Nonempty ↑(F.obj (CategoryTheory.Limits.piObj X))
  -/
  let f (i : ι) : FintypeCat.{w} := F.obj (X i)
  /-
    case intro
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝³ : CategoryTheory.PreGaloisCategory C
    inst✝² : CategoryTheory.PreGaloisCategory.FiberFunctor F
    ι : Type u_1
    inst✝¹ : Finite ι
    X : ι → C
    inst✝ : ∀ (i : ι), Nonempty ↑(F.obj (X i))
    val✝ : Fintype ι
    f : ι → FintypeCat := fun i => F.obj (X i)
    ⊢ Nonempty ↑(F.obj (CategoryTheory.Limits.piObj X))
  -/
  let i : F.obj (∏ᶜ X) ≅ ∏ᶜ f := PreservesProduct.iso F _
  /-
    case intro
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝³ : CategoryTheory.PreGaloisCategory C
    inst✝² : CategoryTheory.PreGaloisCategory.FiberFunctor F
    ι : Type u_1
    inst✝¹ : Finite ι
    X : ι → C
    inst✝ : ∀ (i : ι), Nonempty ↑(F.obj (X i))
    val✝ : Fintype ι
    f : ι → FintypeCat := fun i => F.obj (X i)
    i : CategoryTheory.Iso (F.obj (CategoryTheory.Limits.piObj X)) (CategoryTheory …
    ⊢ Nonempty ↑(F.obj (CategoryTheory.Limits.piObj X))
  -/
  exact Nonempty.elim inferInstance fun x : (∏ᶜ f : FintypeCat.{w}) ↦ ⟨i.inv x⟩
  /-
    🎉 no goals
  -/


/-- A mono between objects with equally sized fibers is an iso. -/
lemma isIso_of_mono_of_eq_card_fiber {X Y : C} (f : X ⟶ Y) [Mono f]
    (h : Nat.card (F.obj X) = Nat.card (F.obj Y)) : IsIso f := by
  have : IsIso (F.map f) := by
    apply (ConcreteCategory.isIso_iff_bijective (F.map f)).mpr
    apply (Fintype.bijective_iff_injective_and_card (F.map f)).mpr
    refine ⟨injective_of_mono_of_preservesPullback (F.map f), ?_⟩
    simp only [← Nat.card_eq_fintype_card, h]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝² : CategoryTheory.PreGaloisCategory C
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Mono f
    h : Eq (Nat.card ↑(F.obj X)) (Nat.card ↑(F.obj Y))
    this : CategoryTheory.IsIso (F.map f)
    ⊢ CategoryTheory.IsIso f
  -/
  exact isIso_of_reflects_iso f F
  /-
    🎉 no goals
  -/


/-- Along a mono that is not an iso, the cardinality of the fiber strictly increases. -/
lemma lt_card_fiber_of_mono_of_notIso {X Y : C} (f : X ⟶ Y) [Mono f]
    (h : ¬ IsIso f) : Nat.card (F.obj X) < Nat.card (F.obj Y) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝² : CategoryTheory.PreGaloisCategory C
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Mono f
    h : Not (CategoryTheory.IsIso f)
    ⊢ LT.lt (Nat.card ↑(F.obj X)) (Nat.card ↑(F.obj Y))
  -/
  by_contra hlt
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝² : CategoryTheory.PreGaloisCategory C
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Mono f
    h : Not (CategoryTheory.IsIso f)
    hlt : Not (LT.lt (Nat.card ↑(F.obj X)) (Nat.card ↑(F.obj Y)))
    ⊢ False
  -/
  apply h
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝² : CategoryTheory.PreGaloisCategory C
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Mono f
    h : Not (CategoryTheory.IsIso f)
    hlt : Not (LT.lt (Nat.card ↑(F.obj X)) (Nat.card ↑(F.obj Y)))
    ⊢ CategoryTheory.IsIso f
  -/
  apply isIso_of_mono_of_eq_card_fiber F f
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝² : CategoryTheory.PreGaloisCategory C
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Mono f
    h : Not (CategoryTheory.IsIso f)
    hlt : Not (LT.lt (Nat.card ↑(F.obj X)) (Nat.card ↑(F.obj Y)))
    ⊢ Eq (Nat.card ↑(F.obj X)) (Nat.card ↑(F.obj Y))
  -/
  simp only [gt_iff_lt, not_lt] at hlt
  exact Nat.le_antisymm
    (Finite.card_le_of_injective (F.map f) (injective_of_mono_of_preservesPullback (F.map f))) hlt


/-- The cardinality of the fiber of a not-initial object is non-zero. -/
lemma non_zero_card_fiber_of_not_initial (X : C) (h : IsInitial X → False) :
    Nat.card (F.obj X) ≠ 0 := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    h : CategoryTheory.Limits.IsInitial X → False
    ⊢ Ne (Nat.card ↑(F.obj X)) 0
  -/
  intro hzero
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    h : CategoryTheory.Limits.IsInitial X → False
    hzero : Eq (Nat.card ↑(F.obj X)) 0
    ⊢ False
  -/
  refine Nonempty.elim ?_ h
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    h : CategoryTheory.Limits.IsInitial X → False
    hzero : Eq (Nat.card ↑(F.obj X)) 0
    ⊢ Nonempty (CategoryTheory.Limits.IsInitial X)
  -/
  rw [initial_iff_fiber_empty F]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    h : CategoryTheory.Limits.IsInitial X → False
    hzero : Eq (Nat.card ↑(F.obj X)) 0
    ⊢ IsEmpty ↑(F.obj X)
  -/
  exact Finite.card_eq_zero_iff.mp hzero
  /-
    🎉 no goals
  -/


/-- The cardinality of the fiber of a coproduct is the sum of the cardinalities of the fibers. -/
lemma card_fiber_coprod_eq_sum (X Y : C) :
    Nat.card (F.obj (X ⨿ Y)) = Nat.card (F.obj X) + Nat.card (F.obj Y) := by
  let e : F.obj (X ⨿ Y) ≃ F.obj X ⊕ F.obj Y := Iso.toEquiv
    <| (PreservesColimitPair.iso (F ⋙ FintypeCat.incl) X Y).symm.trans
    <| Types.binaryCoproductIso (FintypeCat.incl.obj (F.obj X)) (FintypeCat.incl.obj (F.obj Y))
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X Y : C
    e : Equiv (↑(F.obj (CategoryTheory.Limits.coprod X Y))) (Sum ↑(F.obj X) ↑(F.ob …
    ⊢ Eq (Nat.card ↑(F.obj (CategoryTheory.Limits.coprod X Y))) (HAdd.hAdd (Nat.ca …
  -/
  rw [← Nat.card_sum]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X Y : C
    e : Equiv (↑(F.obj (CategoryTheory.Limits.coprod X Y))) (Sum ↑(F.obj X) ↑(F.ob …
    ⊢ Eq (Nat.card ↑(F.obj (CategoryTheory.Limits.coprod X Y))) (Nat.card (Sum ↑(F …
  -/
  exact Nat.card_eq_of_bijective e.toFun (Equiv.bijective e)
  /-
    🎉 no goals
  -/


/-- The cardinality of morphisms `A ⟶ X` is smaller than the cardinality of
the fiber of the target if the source is connected. -/
lemma card_hom_le_card_fiber_of_connected (A X : C) [IsConnected A] :
    Nat.card (A ⟶ X) ≤ Nat.card (F.obj X) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝² : CategoryTheory.PreGaloisCategory C
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    A X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    ⊢ LE.le (Nat.card (Quiver.Hom A X)) (Nat.card ↑(F.obj X))
  -/
  apply Nat.card_le_card_of_injective
  /-
    case hf
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝² : CategoryTheory.PreGaloisCategory C
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    A X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    ⊢ Function.Injective ?f
  -/
  exact evaluation_injective_of_isConnected F A X (Classical.arbitrary _)
  /-
    🎉 no goals
  -/


/-- If `A` is connected, the cardinality of `Aut A` is smaller than the cardinality of the
fiber of `A`. -/
lemma card_aut_le_card_fiber_of_connected (A : C) [IsConnected A] :
    Nat.card (Aut A) ≤ Nat.card (F.obj A) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝² : CategoryTheory.PreGaloisCategory C
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    A : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    ⊢ LE.le (Nat.card (CategoryTheory.Aut A)) (Nat.card ↑(F.obj A))
  -/
  have h : Nonempty (F.obj A) := inferInstance
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝² : CategoryTheory.PreGaloisCategory C
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    A : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    h : Nonempty ↑(F.obj A)
    ⊢ LE.le (Nat.card (CategoryTheory.Aut A)) (Nat.card ↑(F.obj A))
  -/
  obtain ⟨a⟩ := h
  /-
    case intro
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝² : CategoryTheory.PreGaloisCategory C
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    A : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    a : ↑(F.obj A)
    ⊢ LE.le (Nat.card (CategoryTheory.Aut A)) (Nat.card ↑(F.obj A))
  -/
  apply Nat.card_le_card_of_injective
  /-
    case intro.hf
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝² : CategoryTheory.PreGaloisCategory C
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    A : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    a : ↑(F.obj A)
    ⊢ Function.Injective ?intro.f
  -/
  exact evaluation_aut_injective_of_isConnected _ _ a
  /-
    🎉 no goals
  -/


/-- A `PreGaloisCategory` is a `GaloisCategory` if it admits a fiber functor. -/
class GaloisCategory (C : Type u₁) [Category.{u₂, u₁} C]
    extends PreGaloisCategory C : Prop where
  hasFiberFunctor : ∃ F : C ⥤ FintypeCat.{u₂}, Nonempty (PreGaloisCategory.FiberFunctor F)


/-- Arbitrarily choose a fiber functor for a Galois category using choice. -/
noncomputable def GaloisCategory.getFiberFunctor : C ⥤ FintypeCat.{u₂} :=
  Classical.choose <| @GaloisCategory.hasFiberFunctor C _ _


/-- The arbitrarily chosen fiber functor `GaloisCategory.getFiberFunctor` is a fiber functor. -/
noncomputable instance : FiberFunctor (GaloisCategory.getFiberFunctor C) :=
  Classical.choice <| Classical.choose_spec (@GaloisCategory.hasFiberFunctor C _ _)


/-- In a `GaloisCategory` the set of morphisms out of a connected object is finite. -/
instance (A X : C) [IsConnected A] : Finite (A ⟶ X) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    A X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    ⊢ Finite (Quiver.Hom A X)
  -/
  let F := GaloisCategory.getFiberFunctor C
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    A X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    F : CategoryTheory.Functor C FintypeCat := CategoryTheory.PreGaloisCategory.Ga …
    ⊢ Finite (Quiver.Hom A X)
  -/
  obtain ⟨a⟩ := nonempty_fiber_of_isConnected F A
  /-
    case intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    A X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    F : CategoryTheory.Functor C FintypeCat := CategoryTheory.PreGaloisCategory.Ga …
    a : ↑(F.obj A)
    ⊢ Finite (Quiver.Hom A X)
  -/
  apply Finite.of_injective (fun f ↦ F.map f a)
  /-
    case intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    A X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    F : CategoryTheory.Functor C FintypeCat := CategoryTheory.PreGaloisCategory.Ga …
    a : ↑(F.obj A)
    ⊢ Function.Injective fun f => F.map f a
  -/
  exact evaluation_injective_of_isConnected F A X a
  /-
    🎉 no goals
  -/


/-- In a `GaloisCategory` the set of automorphism of a connected object is finite. -/
instance (A : C) [IsConnected A] : Finite (Aut A) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    A : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    ⊢ Finite (CategoryTheory.Aut A)
  -/
  let F := GaloisCategory.getFiberFunctor C
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    A : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    F : CategoryTheory.Functor C FintypeCat := CategoryTheory.PreGaloisCategory.Ga …
    ⊢ Finite (CategoryTheory.Aut A)
  -/
  obtain ⟨a⟩ := nonempty_fiber_of_isConnected F A
  /-
    case intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    A : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    F : CategoryTheory.Functor C FintypeCat := CategoryTheory.PreGaloisCategory.Ga …
    a : ↑(F.obj A)
    ⊢ Finite (CategoryTheory.Aut A)
  -/
  apply Finite.of_injective (fun f ↦ F.map f.hom a)
  /-
    case intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    A : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected A
    F : CategoryTheory.Functor C FintypeCat := CategoryTheory.PreGaloisCategory.Ga …
    a : ↑(F.obj A)
    ⊢ Function.Injective fun f => F.map f.hom a
  -/
  exact evaluation_aut_injective_of_isConnected F A a
  /-
    🎉 no goals
  -/


/-- Coproduct inclusions are monic in Galois categories. -/
instance : MonoCoprod C := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{u₂, u₁} C
    inst✝ : CategoryTheory.GaloisCategory C
    ⊢ CategoryTheory.Limits.MonoCoprod C
  -/
  let F := GaloisCategory.getFiberFunctor C
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{u₂, u₁} C
    inst✝ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat := CategoryTheory.PreGaloisCategory.Ga …
    ⊢ CategoryTheory.Limits.MonoCoprod C
  -/
  exact MonoCoprod.monoCoprod_of_preservesCoprod_of_reflectsMono F
  /-
    🎉 no goals
  -/


