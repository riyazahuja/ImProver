/-- `Spec ℤ` is the terminal object in the category of schemes. -/
noncomputable def specZIsTerminal : IsTerminal (Spec (CommRingCat.of ℤ)) :=
  @IsTerminal.isTerminalObj _ _ _ _ Scheme.Spec _ inferInstance
    (terminalOpOfInitial CommRingCat.zIsInitial)


instance : HasTerminal Scheme :=
  hasTerminal_of_hasTerminal_of_preservesLimit Scheme.Spec


instance : IsAffine (⊤_ Scheme.{u}) :=
  isAffine_of_isIso (PreservesTerminal.iso Scheme.Spec).inv


instance : HasFiniteLimits Scheme :=
  hasFiniteLimits_of_hasTerminal_and_pullbacks


instance (X : Scheme.{u}) : X.Over (⊤_ _) := ⟨terminal.from _⟩

instance {X Y : Scheme.{u}} [X.Over (⊤_ Scheme)] [Y.Over (⊤_ Scheme)] (f : X ⟶ Y) :
    @Scheme.Hom.IsOver _ _ f (⊤_ Scheme) ‹_› ‹_› := ⟨Subsingleton.elim _ _⟩


instance {X : Scheme} : Subsingleton (X.Over (⊤_ Scheme)) :=
                    /-
                      X : AlgebraicGeometry.Scheme
                      x✝¹ x✝ : X.Over (CategoryTheory.Limits.terminal AlgebraicGeometry.Scheme)
                      a b : Quiver.Hom X (CategoryTheory.Limits.terminal AlgebraicGeometry.Scheme)
                      ⊢ Eq { hom := a } { hom := b }
                    -/
  ⟨fun ⟨a⟩ ⟨b⟩ ↦ by simp [Subsingleton.elim a b]⟩
                    /-
                      🎉 no goals
                    -/


/-- The map from the empty scheme. -/
@[simps]
def Scheme.emptyTo (X : Scheme.{u}) : ∅ ⟶ X :=
                                          /-
                                            X : AlgebraicGeometry.Scheme
                                            ⊢ Continuous fun x => PEmpty.elim x
                                          -/
  ⟨{  base := ⟨fun x => PEmpty.elim x, by fun_prop⟩
                                          /-
                                            🎉 no goals
                                          -/
      c := { app := fun _ => CommRingCat.punitIsTerminal.from _ } }, fun x => PEmpty.elim x⟩


@[ext]
theorem Scheme.empty_ext {X : Scheme.{u}} (f g : ∅ ⟶ X) : f = g :=
  Scheme.Hom.ext' (Subsingleton.elim (α := ∅ ⟶ _) _ _)


theorem Scheme.eq_emptyTo {X : Scheme.{u}} (f : ∅ ⟶ X) : f = Scheme.emptyTo X :=
  Scheme.empty_ext f (Scheme.emptyTo X)


instance Scheme.hom_unique_of_empty_source (X : Scheme.{u}) : Unique (∅ ⟶ X) :=
  ⟨⟨Scheme.emptyTo _⟩, fun _ => Scheme.empty_ext _ _⟩


/-- The empty scheme is the initial object in the category of schemes. -/
def emptyIsInitial : IsInitial (∅ : Scheme.{u}) :=
  IsInitial.ofUnique _


@[simp]
theorem emptyIsInitial_to : emptyIsInitial.to = Scheme.emptyTo :=
  rfl


instance : IsEmpty (∅ : Scheme.{u}) :=
                         /-
                           ⊢ IsEmpty PEmpty.{u + 1}
                         -/
  show IsEmpty PEmpty by infer_instance
                         /-
                           🎉 no goals
                         -/


instance spec_punit_isEmpty : IsEmpty (Spec (CommRingCat.of PUnit.{u+1})) :=
  inferInstanceAs <| IsEmpty (PrimeSpectrum PUnit)


instance (priority := 100) isOpenImmersion_of_isEmpty {X Y : Scheme} (f : X ⟶ Y)
    [IsEmpty X] : IsOpenImmersion f := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : IsEmpty ↑↑X.toPresheafedSpace
    ⊢ AlgebraicGeometry.IsOpenImmersion f
  -/
  apply (config := { allowSynthFailures := true }) IsOpenImmersion.of_stalk_iso
    /-
      case hf
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝ : IsEmpty ↑↑X.toPresheafedSpace
      ⊢ Topology.IsOpenEmbedding ⇑f.base
    -/
  · exact .of_isEmpty (X := X) _
    /-
      🎉 no goals
    -/
    /-
      case inst
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝ : IsEmpty ↑↑X.toPresheafedSpace
      ⊢ ∀ (x : ↑↑X.toPresheafedSpace), CategoryTheory.IsIso (AlgebraicGeometry.Schem …
    -/
  · intro (i : X); exact isEmptyElim i
                   /-
                     🎉 no goals
                   -/


instance (priority := 100) isIso_of_isEmpty {X Y : Scheme} (f : X ⟶ Y) [IsEmpty Y] :
    IsIso f := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : IsEmpty ↑↑Y.toPresheafedSpace
    ⊢ CategoryTheory.IsIso f
  -/
  haveI : IsEmpty X := f.base.1.isEmpty
  have : Epi f.base := by
    rw [TopCat.epi_iff_surjective]; rintro (x : Y)
    exact isEmptyElim x
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : IsEmpty ↑↑Y.toPresheafedSpace
    this✝ : IsEmpty ↑↑X.toPresheafedSpace
    this : CategoryTheory.Epi f.base
    ⊢ CategoryTheory.IsIso f
  -/
  apply IsOpenImmersion.to_iso
  /-
    🎉 no goals
  -/


/-- A scheme is initial if its underlying space is empty . -/
noncomputable def isInitialOfIsEmpty {X : Scheme} [IsEmpty X] : IsInitial X :=
  emptyIsInitial.ofIso (asIso <| emptyIsInitial.to _)


/-- `Spec 0` is the initial object in the category of schemes. -/
noncomputable def specPunitIsInitial : IsInitial (Spec (.of PUnit.{u+1})) :=
  emptyIsInitial.ofIso (asIso <| emptyIsInitial.to _)


instance (priority := 100) isAffine_of_isEmpty {X : Scheme} [IsEmpty X] : IsAffine X :=
  isAffine_of_isIso (inv (emptyIsInitial.to X) ≫ emptyIsInitial.to (Spec (.of PUnit)))


instance : HasInitial Scheme.{u} :=
  hasInitial_of_unique ∅


instance initial_isEmpty : IsEmpty (⊥_ Scheme) :=
  ⟨fun x => ((initial.to Scheme.empty : _).base x).elim⟩


theorem isAffineOpen_bot (X : Scheme) : IsAffineOpen (⊥ : X.Opens) :=
  @isAffine_of_isEmpty _ (inferInstanceAs (IsEmpty (∅ : Set X)))


instance : HasStrictInitialObjects Scheme :=
                                                             /-
                                                               A : AlgebraicGeometry.Scheme
                                                               f : Quiver.Hom A (CategoryTheory.Limits.initial AlgebraicGeometry.Scheme)
                                                               ⊢ CategoryTheory.IsIso f
                                                             -/
  hasStrictInitialObjects_of_initial_is_strict fun A f => by infer_instance
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- (Implementation Detail) The glue data associated to a disjoint union. -/
@[simps]
noncomputable
def disjointGlueData' : GlueData' Scheme where
  J := ι
  U := f
  V _ _ _ := ∅
  f _ _ _ := Scheme.emptyTo _
  t _ _ _ := 𝟙 _
  t' _ _ _ _ _ _ := Limits.pullback.fst _ _ ≫ Scheme.emptyTo _
  t_fac _ _ _ _ _ _ := emptyIsInitial.strict_hom_ext _ _
  t_inv _ _ _ := Category.comp_id _
  cocycle _ _ _ _ _ _ := (emptyIsInitial.ofStrict (pullback.fst _ _)).hom_ext _ _
                   /-
                     ι : Type u
                     f : ι → AlgebraicGeometry.Scheme
                     x✝¹ x✝ : ι
                     ⊢ ∀ (h : Ne x✝¹ x✝), CategoryTheory.Mono ((fun x x_1 x_2 => (f x).emptyTo) x✝¹ …
                   -/
  f_mono _ _ := by dsimp only; infer_instance
                               /-
                                 🎉 no goals
                               -/


/-- (Implementation Detail) The glue data associated to a disjoint union. -/
@[simps! J V U f t]
noncomputable
def disjointGlueData : Scheme.GlueData where
  __ := GlueData.ofGlueData' (disjointGlueData' f)
  f_open i j := by
    /-
      ι : Type u
      f : ι → AlgebraicGeometry.Scheme
      i j : __spread✝⁻⁰.J
      ⊢ AlgebraicGeometry.IsOpenImmersion (__spread✝⁻⁰.f i j)
    -/
    dsimp only [GlueData.ofGlueData', GlueData'.f', disjointGlueData']
    /-
      ι : Type u
      f : ι → AlgebraicGeometry.Scheme
      i j : __spread✝⁻⁰.J
      ⊢ AlgebraicGeometry.IsOpenImmersion (dite (Eq i j) (fun h => CategoryTheory.eq …
    -/
              /-
                🎉 no goals
              -/
    split <;> infer_instance
              /-
                🎉 no goals
              -/


/-- (Implementation Detail) The cofan in `LocallyRingedSpace` associated to a disjoint union. -/
noncomputable
def toLocallyRingedSpaceCoproductCofan : Cofan (Scheme.toLocallyRingedSpace ∘ f) :=
  Cofan.mk (disjointGlueData f).toLocallyRingedSpaceGlueData.glued
    (disjointGlueData f).toLocallyRingedSpaceGlueData.ι


/-- (Implementation Detail)
The cofan in `LocallyRingedSpace` associated to a disjoint union is a colimit. -/
noncomputable
def toLocallyRingedSpaceCoproductCofanIsColimit :
    IsColimit (toLocallyRingedSpaceCoproductCofan f) := by
  /-
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    ⊢ CategoryTheory.Limits.IsColimit (AlgebraicGeometry.toLocallyRingedSpaceCopro …
  -/
  fapply mkCofanColimit
    /-
      case desc
      ι : Type u
      f : ι → AlgebraicGeometry.Scheme
      ⊢ (t : CategoryTheory.Limits.Cofan (Function.comp AlgebraicGeometry.Scheme.toL …
    -/
  · refine fun t ↦ Multicoequalizer.desc _ _ t.inj ?_
    /-
      case desc
      ι : Type u
      f : ι → AlgebraicGeometry.Scheme
      t : CategoryTheory.Limits.Cofan (Function.comp AlgebraicGeometry.Scheme.toLoca …
      ⊢ ∀ (a : (AlgebraicGeometry.disjointGlueData f).toLocallyRingedSpaceGlueData.d …
    -/
    rintro ⟨i, j⟩
    simp only [GlueData.diagram, disjointGlueData_J, disjointGlueData_V, disjointGlueData_U,
      disjointGlueData_f, disjointGlueData_t, Category.comp_id, Category.assoc,
      GlueData.mapGlueData_J, disjointGlueData_J, GlueData.mapGlueData_V,
      disjointGlueData_V, Scheme.forgetToLocallyRingedSpace_obj, GlueData.mapGlueData_U,
      disjointGlueData_U, GlueData.mapGlueData_f, disjointGlueData_f, Category.comp_id,
      Scheme.forgetToLocallyRingedSpace_map, GlueData.mapGlueData_t, disjointGlueData_t]
    /-
      case desc.mk
      ι : Type u
      f : ι → AlgebraicGeometry.Scheme
      t : CategoryTheory.Limits.Cofan (Function.comp AlgebraicGeometry.Scheme.toLoca …
      i j : (AlgebraicGeometry.disjointGlueData f).toLocallyRingedSpaceGlueData.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.toLRSHo …
    -/
    split_ifs with h
      /-
        case pos
        ι : Type u
        f : ι → AlgebraicGeometry.Scheme
        t : CategoryTheory.Limits.Cofan (Function.comp AlgebraicGeometry.Scheme.toLoca …
        i j : (AlgebraicGeometry.disjointGlueData f).toLocallyRingedSpaceGlueData.J
        h : Eq i j
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.toLRSHo …
      -/
    · subst h
      /-
        case pos
        ι : Type u
        f : ι → AlgebraicGeometry.Scheme
        t : CategoryTheory.Limits.Cofan (Function.comp AlgebraicGeometry.Scheme.toLoca …
        i : (AlgebraicGeometry.disjointGlueData f).toLocallyRingedSpaceGlueData.J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.toLRSHo …
      -/
      simp only [eqToHom_refl, ↓reduceDIte, ← Category.assoc, GlueData'.f']
      /-
        case pos
        ι : Type u
        f : ι → AlgebraicGeometry.Scheme
        t : CategoryTheory.Limits.Cofan (Function.comp AlgebraicGeometry.Scheme.toLoca …
        i : (AlgebraicGeometry.disjointGlueData f).toLocallyRingedSpaceGlueData.J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.toLRSHo …
      -/
      congr
      /-
        🎉 no goals
      -/
      /-
        case neg
        ι : Type u
        f : ι → AlgebraicGeometry.Scheme
        t : CategoryTheory.Limits.Cofan (Function.comp AlgebraicGeometry.Scheme.toLoca …
        i j : (AlgebraicGeometry.disjointGlueData f).toLocallyRingedSpaceGlueData.J
        h : Not (Eq i j)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.toLRSHo …
      -/
    · apply Limits.IsInitial.hom_ext
      /-
        case neg.t
        ι : Type u
        f : ι → AlgebraicGeometry.Scheme
        t : CategoryTheory.Limits.Cofan (Function.comp AlgebraicGeometry.Scheme.toLoca …
        i j : (AlgebraicGeometry.disjointGlueData f).toLocallyRingedSpaceGlueData.J
        h : Not (Eq i j)
        ⊢ CategoryTheory.Limits.IsInitial (ite (Eq i j) (f i) EmptyCollection.emptyCol …
      -/
      rw [if_neg h]
      /-
        case neg.t
        ι : Type u
        f : ι → AlgebraicGeometry.Scheme
        t : CategoryTheory.Limits.Cofan (Function.comp AlgebraicGeometry.Scheme.toLoca …
        i j : (AlgebraicGeometry.disjointGlueData f).toLocallyRingedSpaceGlueData.J
        h : Not (Eq i j)
        ⊢ CategoryTheory.Limits.IsInitial EmptyCollection.emptyCollection.toLocallyRin …
      -/
      exact LocallyRingedSpace.emptyIsInitial
      /-
        🎉 no goals
      -/
    /-
      case fac
      ι : Type u
      f : ι → AlgebraicGeometry.Scheme
      ⊢ autoParam (∀ (t : CategoryTheory.Limits.Cofan (Function.comp AlgebraicGeomet …
    -/
  · exact fun _ _ ↦ Multicoequalizer.π_desc _ _ _ _ _
    /-
      🎉 no goals
    -/
    /-
      case uniq
      ι : Type u
      f : ι → AlgebraicGeometry.Scheme
      ⊢ autoParam (∀ (t : CategoryTheory.Limits.Cofan (Function.comp AlgebraicGeomet …
    -/
  · intro i m h
    /-
      case uniq
      ι : Type u
      f : ι → AlgebraicGeometry.Scheme
      i : CategoryTheory.Limits.Cofan (Function.comp AlgebraicGeometry.Scheme.toLoca …
      m : Quiver.Hom (AlgebraicGeometry.toLocallyRingedSpaceCoproductCofan f).pt i.pt
      h : ∀ (j : ι), Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.toLo …
      ⊢ Eq m (CategoryTheory.Limits.Multicoequalizer.desc (AlgebraicGeometry.disjoin …
    -/
    apply Multicoequalizer.hom_ext _ _ _ fun j ↦ ?_
    /-
      ι : Type u
      f : ι → AlgebraicGeometry.Scheme
      i : CategoryTheory.Limits.Cofan (Function.comp AlgebraicGeometry.Scheme.toLoca …
      m : Quiver.Hom (AlgebraicGeometry.toLocallyRingedSpaceCoproductCofan f).pt i.pt
      h : ∀ (j : ι), Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.toLo …
      j : (AlgebraicGeometry.disjointGlueData f).toLocallyRingedSpaceGlueData.diagra …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Multicoequaliz …
    -/
    rw [Multicoequalizer.π_desc]
    /-
      ι : Type u
      f : ι → AlgebraicGeometry.Scheme
      i : CategoryTheory.Limits.Cofan (Function.comp AlgebraicGeometry.Scheme.toLoca …
      m : Quiver.Hom (AlgebraicGeometry.toLocallyRingedSpaceCoproductCofan f).pt i.pt
      h : ∀ (j : ι), Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.toLo …
      j : (AlgebraicGeometry.disjointGlueData f).toLocallyRingedSpaceGlueData.diagra …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Multicoequaliz …
    -/
    exact h j
    /-
      🎉 no goals
    -/


noncomputable
instance : CreatesColimit (Discrete.functor f) Scheme.forgetToLocallyRingedSpace :=
  createsColimitOfFullyFaithfulOfIso (disjointGlueData f).gluedScheme <|
    let F : Discrete.functor f ⋙ Scheme.forgetToLocallyRingedSpace ≅
      Discrete.functor (Scheme.toLocallyRingedSpace ∘ f) := Discrete.natIsoFunctor
    have := (IsColimit.precomposeHomEquiv F _).symm (toLocallyRingedSpaceCoproductCofanIsColimit f)
    (colimit.isoColimitCocone ⟨_, this⟩).symm


noncomputable
instance : CreatesColimitsOfShape (Discrete ι) Scheme.forgetToLocallyRingedSpace := by
  /-
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    ⊢ CategoryTheory.CreatesColimitsOfShape (CategoryTheory.Discrete ι) AlgebraicG …
  -/
  constructor
  /-
    case CreatesColimit
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    ⊢ autoParam ({K : CategoryTheory.Functor (CategoryTheory.Discrete ι) Algebraic …
  -/
  intro K
  /-
    case CreatesColimit
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    K : CategoryTheory.Functor (CategoryTheory.Discrete ι) AlgebraicGeometry.Scheme
    ⊢ CategoryTheory.CreatesColimit K AlgebraicGeometry.Scheme.forgetToLocallyRing …
  -/
  exact createsColimitOfIsoDiagram _ (Discrete.natIsoFunctor (F := K)).symm
  /-
    🎉 no goals
  -/


instance : PreservesColimitsOfShape (Discrete ι) Scheme.forgetToTop.{u} :=
  inferInstanceAs (PreservesColimitsOfShape (Discrete ι) (Scheme.forgetToLocallyRingedSpace ⋙
      LocallyRingedSpace.forgetToSheafedSpace ⋙ SheafedSpace.forget CommRingCat))


instance : HasCoproducts.{u} Scheme.{u} :=
  fun _ ↦ ⟨fun _ ↦ hasColimit_of_created _ Scheme.forgetToLocallyRingedSpace⟩


instance : HasCoproducts.{0} Scheme.{u} := has_smallest_coproducts_of_hasCoproducts


noncomputable
instance {ι : Type} : PreservesColimitsOfShape (Discrete ι) Scheme.forgetToTop.{u} :=
  preservesColimitsOfShape_of_equiv
    (Discrete.equivalence Equiv.ulift : Discrete (ULift.{u} ι) ≌ _) _


noncomputable
instance {ι : Type} :
    PreservesColimitsOfShape (Discrete ι) Scheme.forgetToLocallyRingedSpace.{u} :=
  preservesColimitsOfShape_of_equiv
    (Discrete.equivalence Equiv.ulift : Discrete (ULift.{u} ι) ≌ _) _


/-- (Implementation Detail) Coproduct of schemes is isomorphic to the disjoint union. -/
noncomputable
def sigmaIsoGlued : ∐ f ≅ (disjointGlueData f).glued :=
  Scheme.fullyFaithfulForgetToLocallyRingedSpace.preimageIso
    (PreservesCoproduct.iso _ _ ≪≫
      colimit.isoColimitCocone ⟨_, toLocallyRingedSpaceCoproductCofanIsColimit f⟩ ≪≫
        (disjointGlueData f).isoLocallyRingedSpace.symm)


@[reassoc (attr := simp)]
lemma ι_sigmaIsoGlued_inv (i) : (disjointGlueData f).ι i ≫ (sigmaIsoGlued f).inv = Sigma.ι f i := by
  /-
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    i : (AlgebraicGeometry.disjointGlueData f).J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.disjointGlueData  …
  -/
  apply Scheme.forgetToLocallyRingedSpace.map_injective
  /-
    case a
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    i : (AlgebraicGeometry.disjointGlueData f).J
    ⊢ Eq (AlgebraicGeometry.Scheme.forgetToLocallyRingedSpace.map (CategoryTheory. …
  -/
  dsimp [sigmaIsoGlued]
  /-
    case a
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    i : (AlgebraicGeometry.disjointGlueData f).J
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.toLRSHom (CategoryTheory.CategoryStruct.com …
  -/
  simp only [Category.assoc]
  /-
    case a
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    i : (AlgebraicGeometry.disjointGlueData f).J
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.toLRSHom (CategoryTheory.CategoryStruct.com …
  -/
  refine ((disjointGlueData f).ι_gluedIso_hom_assoc Scheme.forgetToLocallyRingedSpace i _).trans ?_
  refine (colimit.isoColimitCocone_ι_inv_assoc
    ⟨_, toLocallyRingedSpaceCoproductCofanIsColimit f⟩ ⟨i⟩ _).trans ?_
  /-
    case a
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    i : (AlgebraicGeometry.disjointGlueData f).J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (Cat …
  -/
  exact ι_comp_sigmaComparison Scheme.forgetToLocallyRingedSpace _ _
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma ι_sigmaIsoGlued_hom (i) :
    Sigma.ι f i ≫ (sigmaIsoGlued f).hom = (disjointGlueData f).ι i := by
  /-
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    i : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι f i) ( …
  -/
  rw [← ι_sigmaIsoGlued_inv, Category.assoc, Iso.inv_hom_id, Category.comp_id]
  /-
    🎉 no goals
  -/


instance (i) : IsOpenImmersion (Sigma.ι f i) := by
  /-
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    i : ι
    ⊢ AlgebraicGeometry.IsOpenImmersion (CategoryTheory.Limits.Sigma.ι f i)
  -/
  rw [← ι_sigmaIsoGlued_inv]
  /-
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    i : ι
    ⊢ AlgebraicGeometry.IsOpenImmersion (CategoryTheory.CategoryStruct.comp ((Alge …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma sigmaι_eq_iff (i j : ι) (x y) :
    (Sigma.ι f i).base x = (Sigma.ι f j).base y ↔
      (Sigma.mk i x : Σ i, f i) = Sigma.mk j y := by
  /-
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    i j : ι
    x : ↑↑(f i).toPresheafedSpace
    y : ↑↑(f j).toPresheafedSpace
    ⊢ Iff (Eq ((CategoryTheory.Limits.Sigma.ι f i).base x) ((CategoryTheory.Limits …
  -/
  constructor
    /-
      case mp
      ι : Type u
      f : ι → AlgebraicGeometry.Scheme
      i j : ι
      x : ↑↑(f i).toPresheafedSpace
      y : ↑↑(f j).toPresheafedSpace
      ⊢ Eq ((CategoryTheory.Limits.Sigma.ι f i).base x) ((CategoryTheory.Limits.Sigm …
    -/
  · intro H
    /-
      case mp
      ι : Type u
      f : ι → AlgebraicGeometry.Scheme
      i j : ι
      x : ↑↑(f i).toPresheafedSpace
      y : ↑↑(f j).toPresheafedSpace
      H : Eq ((CategoryTheory.Limits.Sigma.ι f i).base x) ((CategoryTheory.Limits.Si …
      ⊢ Eq ⟨i, x⟩ ⟨j, y⟩
    -/
    rw [← ι_sigmaIsoGlued_inv, ← ι_sigmaIsoGlued_inv] at H
    erw [(TopCat.homeoOfIso
      (Scheme.forgetToTop.mapIso (sigmaIsoGlued f))).symm.injective.eq_iff] at H
    /-
      case mp
      ι : Type u
      f : ι → AlgebraicGeometry.Scheme
      i j : ι
      x : ↑↑(f i).toPresheafedSpace
      y : ↑↑(f j).toPresheafedSpace
      H : Eq ((AlgebraicGeometry.LocallyRingedSpace.Hom.toShHom (AlgebraicGeometry.S …
      ⊢ Eq ⟨i, x⟩ ⟨j, y⟩
    -/
    by_cases h : i = j
      /-
        case pos
        ι : Type u
        f : ι → AlgebraicGeometry.Scheme
        i j : ι
        x : ↑↑(f i).toPresheafedSpace
        y : ↑↑(f j).toPresheafedSpace
        H : Eq ((AlgebraicGeometry.LocallyRingedSpace.Hom.toShHom (AlgebraicGeometry.S …
        h : Eq i j
        ⊢ Eq ⟨i, x⟩ ⟨j, y⟩
      -/
    · subst h
      /-
        case pos
        ι : Type u
        f : ι → AlgebraicGeometry.Scheme
        i : ι
        x y : ↑↑(f i).toPresheafedSpace
        H : Eq ((AlgebraicGeometry.LocallyRingedSpace.Hom.toShHom (AlgebraicGeometry.S …
        ⊢ Eq ⟨i, x⟩ ⟨i, y⟩
      -/
      simp only [Sigma.mk.inj_iff, heq_eq_eq, true_and]
      /-
        case pos
        ι : Type u
        f : ι → AlgebraicGeometry.Scheme
        i : ι
        x y : ↑↑(f i).toPresheafedSpace
        H : Eq ((AlgebraicGeometry.LocallyRingedSpace.Hom.toShHom (AlgebraicGeometry.S …
        ⊢ Eq x y
      -/
      exact ((disjointGlueData f).ι i).isOpenEmbedding.injective H
      /-
        🎉 no goals
      -/
      /-
        case neg
        ι : Type u
        f : ι → AlgebraicGeometry.Scheme
        i j : ι
        x : ↑↑(f i).toPresheafedSpace
        y : ↑↑(f j).toPresheafedSpace
        H : Eq ((AlgebraicGeometry.LocallyRingedSpace.Hom.toShHom (AlgebraicGeometry.S …
        h : Not (Eq i j)
        ⊢ Eq ⟨i, x⟩ ⟨j, y⟩
      -/
    · obtain (e | ⟨z, _⟩) := (Scheme.GlueData.ι_eq_iff _ _ _ _ _).mp H
        /-
          case neg.inl
          ι : Type u
          f : ι → AlgebraicGeometry.Scheme
          i j : ι
          x : ↑↑(f i).toPresheafedSpace
          y : ↑↑(f j).toPresheafedSpace
          H : Eq ((AlgebraicGeometry.LocallyRingedSpace.Hom.toShHom (AlgebraicGeometry.S …
          h : Not (Eq i j)
          e : Eq ⟨i, x⟩ ⟨j, y⟩
          ⊢ Eq ⟨i, x⟩ ⟨j, y⟩
        -/
      · exact (h (Sigma.mk.inj_iff.mp e).1).elim
        /-
          🎉 no goals
        -/
        /-
          case neg.inr.intro
          ι : Type u
          f : ι → AlgebraicGeometry.Scheme
          i j : ι
          x : ↑↑(f i).toPresheafedSpace
          y : ↑↑(f j).toPresheafedSpace
          H : Eq ((AlgebraicGeometry.LocallyRingedSpace.Hom.toShHom (AlgebraicGeometry.S …
          h : Not (Eq i j)
          z : ↑↑((AlgebraicGeometry.disjointGlueData f).V { fst := ⟨i, x⟩.fst, snd := ⟨j …
          h✝ : And (Eq (((AlgebraicGeometry.disjointGlueData f).f ⟨i, x⟩.fst ⟨j, y⟩.fst) …
          ⊢ Eq ⟨i, x⟩ ⟨j, y⟩
        -/
      · simp only [disjointGlueData_J, disjointGlueData_V, h, ↓reduceIte] at z
        /-
          case neg.inr.intro
          ι : Type u
          f : ι → AlgebraicGeometry.Scheme
          i j : ι
          x : ↑↑(f i).toPresheafedSpace
          y : ↑↑(f j).toPresheafedSpace
          H : Eq ((AlgebraicGeometry.LocallyRingedSpace.Hom.toShHom (AlgebraicGeometry.S …
          h : Not (Eq i j)
          z✝ : ↑↑((AlgebraicGeometry.disjointGlueData f).V { fst := ⟨i, x⟩.fst, snd := ⟨ …
          h✝ : And (Eq (((AlgebraicGeometry.disjointGlueData f).f ⟨i, x⟩.fst ⟨j, y⟩.fst) …
          z : ↑↑EmptyCollection.emptyCollection.toPresheafedSpace
          ⊢ Eq ⟨i, x⟩ ⟨j, y⟩
        -/
        cases z
        /-
          🎉 no goals
        -/
    /-
      case mpr
      ι : Type u
      f : ι → AlgebraicGeometry.Scheme
      i j : ι
      x : ↑↑(f i).toPresheafedSpace
      y : ↑↑(f j).toPresheafedSpace
      ⊢ Eq ⟨i, x⟩ ⟨j, y⟩ → Eq ((CategoryTheory.Limits.Sigma.ι f i).base x) ((Categor …
    -/
  · rintro ⟨rfl⟩
    /-
      case mpr.refl
      ι : Type u
      f : ι → AlgebraicGeometry.Scheme
      i : ι
      x : ↑↑(f i).toPresheafedSpace
      ⊢ Eq ((CategoryTheory.Limits.Sigma.ι f i).base x) ((CategoryTheory.Limits.Sigm …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- The images of each component in the coproduct is disjoint. -/
lemma disjoint_opensRange_sigmaι (i j : ι) (h : i ≠ j) :
    Disjoint (Sigma.ι f i).opensRange (Sigma.ι f j).opensRange := by
  /-
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    i j : ι
    h : Ne i j
    ⊢ Disjoint (AlgebraicGeometry.Scheme.Hom.opensRange (CategoryTheory.Limits.Sig …
  -/
  intro U hU hU' x hx
  /-
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    i j : ι
    h : Ne i j
    U : (CategoryTheory.Limits.sigmaObj f).Opens
    hU : LE.le U (AlgebraicGeometry.Scheme.Hom.opensRange (CategoryTheory.Limits.S …
    hU' : LE.le U (AlgebraicGeometry.Scheme.Hom.opensRange (CategoryTheory.Limits. …
    x : ↑↑(CategoryTheory.Limits.sigmaObj f).toPresheafedSpace
    hx : Membership.mem (↑U) x
    ⊢ Membership.mem (↑Bot.bot) x
  -/
  obtain ⟨x, rfl⟩ := hU hx
  /-
    case intro
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    i j : ι
    h : Ne i j
    U : (CategoryTheory.Limits.sigmaObj f).Opens
    hU : LE.le U (AlgebraicGeometry.Scheme.Hom.opensRange (CategoryTheory.Limits.S …
    hU' : LE.le U (AlgebraicGeometry.Scheme.Hom.opensRange (CategoryTheory.Limits. …
    x : ↑↑(f i).toPresheafedSpace
    hx : Membership.mem (↑U) ((CategoryTheory.Limits.Sigma.ι f i).base x)
    ⊢ Membership.mem (↑Bot.bot) ((CategoryTheory.Limits.Sigma.ι f i).base x)
  -/
  obtain ⟨y, hy⟩ := hU' hx
  /-
    case intro.intro
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    i j : ι
    h : Ne i j
    U : (CategoryTheory.Limits.sigmaObj f).Opens
    hU : LE.le U (AlgebraicGeometry.Scheme.Hom.opensRange (CategoryTheory.Limits.S …
    hU' : LE.le U (AlgebraicGeometry.Scheme.Hom.opensRange (CategoryTheory.Limits. …
    x : ↑↑(f i).toPresheafedSpace
    hx : Membership.mem (↑U) ((CategoryTheory.Limits.Sigma.ι f i).base x)
    y : ↑↑(f j).toPresheafedSpace
    hy : Eq ((CategoryTheory.Limits.Sigma.ι f j).base y) ((CategoryTheory.Limits.S …
    ⊢ Membership.mem (↑Bot.bot) ((CategoryTheory.Limits.Sigma.ι f i).base x)
  -/
  obtain ⟨rfl⟩ := (sigmaι_eq_iff _ _ _ _ _).mp hy
  /-
    case intro.intro.refl
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    i : ι
    U : (CategoryTheory.Limits.sigmaObj f).Opens
    hU : LE.le U (AlgebraicGeometry.Scheme.Hom.opensRange (CategoryTheory.Limits.S …
    x : ↑↑(f i).toPresheafedSpace
    hx : Membership.mem (↑U) ((CategoryTheory.Limits.Sigma.ι f i).base x)
    h : Ne i i
    hU' : LE.le U (AlgebraicGeometry.Scheme.Hom.opensRange (CategoryTheory.Limits. …
    hy : Eq ((CategoryTheory.Limits.Sigma.ι f i).base x) ((CategoryTheory.Limits.S …
    ⊢ Membership.mem (↑Bot.bot) ((CategoryTheory.Limits.Sigma.ι f i).base x)
  -/
  cases h rfl
  /-
    🎉 no goals
  -/


lemma exists_sigmaι_eq (x : (∐ f : _)) : ∃ i y, (Sigma.ι f i).base y = x := by
  /-
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    x : ↑↑(CategoryTheory.Limits.sigmaObj f).toPresheafedSpace
    ⊢ Exists fun i => Exists fun y => Eq ((CategoryTheory.Limits.Sigma.ι f i).base …
  -/
  obtain ⟨i, y, e⟩ := (disjointGlueData f).ι_jointly_surjective ((sigmaIsoGlued f).hom.base x)
  /-
    case intro.intro
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    x : ↑↑(CategoryTheory.Limits.sigmaObj f).toPresheafedSpace
    i : (AlgebraicGeometry.disjointGlueData f).J
    y : ↑↑((AlgebraicGeometry.disjointGlueData f).U i).toPresheafedSpace
    e : Eq (((AlgebraicGeometry.disjointGlueData f).ι i).base y) ((AlgebraicGeomet …
    ⊢ Exists fun i => Exists fun y => Eq ((CategoryTheory.Limits.Sigma.ι f i).base …
  -/
  refine ⟨i, y, (sigmaIsoGlued f).hom.isOpenEmbedding.injective ?_⟩
  /-
    case intro.intro
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    x : ↑↑(CategoryTheory.Limits.sigmaObj f).toPresheafedSpace
    i : (AlgebraicGeometry.disjointGlueData f).J
    y : ↑↑((AlgebraicGeometry.disjointGlueData f).U i).toPresheafedSpace
    e : Eq (((AlgebraicGeometry.disjointGlueData f).ι i).base y) ((AlgebraicGeomet …
    ⊢ Eq ((AlgebraicGeometry.sigmaIsoGlued f).hom.base ((CategoryTheory.Limits.Sig …
  -/
  rwa [← Scheme.comp_base_apply, ι_sigmaIsoGlued_hom]
  /-
    🎉 no goals
  -/


lemma iSup_opensRange_sigmaι : ⨆ i, (Sigma.ι f i).opensRange = ⊤ :=
                            /-
                              ι : Type u
                              f : ι → AlgebraicGeometry.Scheme
                              x : ↑↑(CategoryTheory.Limits.sigmaObj f).toPresheafedSpace
                              ⊢ Membership.mem (↑Top.top) x → Membership.mem (↑(iSup fun i => AlgebraicGeome …
                            -/
  eq_top_iff.mpr fun x ↦ by simpa using exists_sigmaι_eq f x
                            /-
                              🎉 no goals
                            -/


/-- The open cover of the coproduct. -/
@[simps obj map]
noncomputable
def sigmaOpenCover : (∐ f).OpenCover where
  J := ι
  obj := f
  map := Sigma.ι f
  f x := (exists_sigmaι_eq f x).choose
  covers x := (exists_sigmaι_eq f x).choose_spec


/-- The underlying topological space of the coproduct is homeomorphic to the disjoint union. -/
noncomputable
def sigmaMk : (Σ i, f i) ≃ₜ (∐ f : _) :=
  TopCat.homeoOfIso ((colimit.isoColimitCocone ⟨_, TopCat.sigmaCofanIsColimit _⟩).symm ≪≫
    (PreservesCoproduct.iso Scheme.forgetToTop f).symm)


@[simp]
lemma sigmaMk_mk (i) (x : f i) :
    sigmaMk f (.mk i x) = (Sigma.ι f i).base x := by
  show ((TopCat.sigmaCofan (fun x ↦ (f x).toTopCat)).inj i ≫
    (colimit.isoColimitCocone ⟨_, TopCat.sigmaCofanIsColimit _⟩).inv ≫ _) x =
      Scheme.forgetToTop.map (Sigma.ι f i) x
  /-
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    i : ι
    x : ↑↑(f i).toPresheafedSpace
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((TopCat.sigmaCofan fun x => (f x).t …
  -/
  congr 1
  /-
    case e_a
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    i : ι
    x : ↑↑(f i).toPresheafedSpace
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((TopCat.sigmaCofan fun x => (f x).to …
  -/
  refine (colimit.isoColimitCocone_ι_inv_assoc ⟨_, TopCat.sigmaCofanIsColimit _⟩ _ _).trans ?_
  /-
    case e_a
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    i : ι
    x : ↑↑(f i).toPresheafedSpace
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (Cat …
  -/
  exact ι_comp_sigmaComparison Scheme.forgetToTop _ _
  /-
    🎉 no goals
  -/


/-- (Implementation Detail)
The coproduct of the two schemes is given by indexed coproducts over `WalkingPair`. -/
noncomputable
def coprodIsoSigma : X ⨿ Y ≅ ∐ fun i : ULift.{u} WalkingPair ↦ i.1.casesOn X Y :=
                                                  /-
                                                    ι : Type u
                                                    f : ι → AlgebraicGeometry.Scheme
                                                    X Y : AlgebraicGeometry.Scheme
                                                    x✝ : CategoryTheory.Limits.WalkingPair
                                                    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.WalkingPair.casesOn (Equiv.ulift.s …
                                                  -/
  Sigma.whiskerEquiv Equiv.ulift.symm (fun _ ↦ by exact Iso.refl _)
                                                  /-
                                                    🎉 no goals
                                                  -/


lemma ι_left_coprodIsoSigma_inv : Sigma.ι _ ⟨.left⟩ ≫ (coprodIsoSigma X Y).inv = coprod.inl :=
  Sigma.ι_comp_map' _ _ _


lemma ι_right_coprodIsoSigma_inv : Sigma.ι _ ⟨.right⟩ ≫ (coprodIsoSigma X Y).inv = coprod.inr :=
  Sigma.ι_comp_map' _ _ _


instance : IsOpenImmersion (coprod.inl : X ⟶ X ⨿ Y) := by
  /-
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    X Y : AlgebraicGeometry.Scheme
    ⊢ AlgebraicGeometry.IsOpenImmersion CategoryTheory.Limits.coprod.inl
  -/
  rw [← ι_left_coprodIsoSigma_inv]; infer_instance
                                    /-
                                      🎉 no goals
                                    -/


instance : IsOpenImmersion (coprod.inr : Y ⟶ X ⨿ Y) := by
  /-
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    X Y : AlgebraicGeometry.Scheme
    ⊢ AlgebraicGeometry.IsOpenImmersion CategoryTheory.Limits.coprod.inr
  -/
  rw [← ι_right_coprodIsoSigma_inv]; infer_instance
                                     /-
                                       🎉 no goals
                                     -/


lemma isCompl_range_inl_inr :
    IsCompl (Set.range (coprod.inl : X ⟶ X ⨿ Y).base)
      (Set.range (coprod.inr : Y ⟶ X ⨿ Y).base) :=
  ((TopCat.binaryCofan_isColimit_iff _).mp
    ⟨mapIsColimitOfPreservesOfIsColimit Scheme.forgetToTop _ _ (coprodIsCoprod X Y)⟩).2.2


lemma isCompl_opensRange_inl_inr :
    IsCompl (coprod.inl : X ⟶ X ⨿ Y).opensRange (coprod.inr : Y ⟶ X ⨿ Y).opensRange := by
  /-
    X Y : AlgebraicGeometry.Scheme
    ⊢ IsCompl (AlgebraicGeometry.Scheme.Hom.opensRange CategoryTheory.Limits.copro …
  -/
  convert isCompl_range_inl_inr X Y
  /-
    case a
    X Y : AlgebraicGeometry.Scheme
    ⊢ Iff (IsCompl (AlgebraicGeometry.Scheme.Hom.opensRange CategoryTheory.Limits. …
  -/
  simp only [isCompl_iff, disjoint_iff, codisjoint_iff, ← TopologicalSpace.Opens.coe_inj]
  /-
    case a
    X Y : AlgebraicGeometry.Scheme
    ⊢ Iff (And (Eq ↑(Min.min (AlgebraicGeometry.Scheme.Hom.opensRange CategoryTheo …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The underlying topological space of the coproduct is homeomorphic to the disjoint union -/
noncomputable
def coprodMk : X ⊕ Y ≃ₜ (X ⨿ Y : Scheme.{u}) :=
  TopCat.homeoOfIso ((colimit.isoColimitCocone ⟨_, TopCat.binaryCofanIsColimit _ _⟩).symm ≪≫
    PreservesColimitPair.iso Scheme.forgetToTop X Y)


@[simp]
lemma coprodMk_inl (x : X) :
    coprodMk X Y (.inl x) = (coprod.inl : X ⟶ X ⨿ Y).base x := by
  show ((TopCat.binaryCofan X Y).inl ≫
    (colimit.isoColimitCocone ⟨_, TopCat.binaryCofanIsColimit _ _⟩).inv ≫ _) x =
      Scheme.forgetToTop.map coprod.inl x
  /-
    X Y : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((↑X.toPresheafedSpace).binaryCofan  …
  -/
  congr 1
  /-
    case e_a
    X Y : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((↑X.toPresheafedSpace).binaryCofan ↑ …
  -/
  refine (colimit.isoColimitCocone_ι_inv_assoc ⟨_, TopCat.binaryCofanIsColimit _ _⟩ _ _).trans ?_
  /-
    case e_a
    X Y : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (Cat …
  -/
  exact coprodComparison_inl Scheme.forgetToTop
  /-
    🎉 no goals
  -/


@[simp]
lemma coprodMk_inr (x : Y) :
    coprodMk X Y (.inr x) = (coprod.inr : Y ⟶ X ⨿ Y).base x := by
  show ((TopCat.binaryCofan X Y).inr ≫
    (colimit.isoColimitCocone ⟨_, TopCat.binaryCofanIsColimit _ _⟩).inv ≫ _) x =
      Scheme.forgetToTop.map coprod.inr x
  /-
    X Y : AlgebraicGeometry.Scheme
    x : ↑↑Y.toPresheafedSpace
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((↑X.toPresheafedSpace).binaryCofan  …
  -/
  congr 1
  /-
    case e_a
    X Y : AlgebraicGeometry.Scheme
    x : ↑↑Y.toPresheafedSpace
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((↑X.toPresheafedSpace).binaryCofan ↑ …
  -/
  refine (colimit.isoColimitCocone_ι_inv_assoc ⟨_, TopCat.binaryCofanIsColimit _ _⟩ _ _).trans ?_
  /-
    case e_a
    X Y : AlgebraicGeometry.Scheme
    x : ↑↑Y.toPresheafedSpace
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (Cat …
  -/
  exact coprodComparison_inr Scheme.forgetToTop
  /-
    🎉 no goals
  -/


/-- The open cover of the coproduct of two schemes. -/
noncomputable
def coprodOpenCover.{w} : (X ⨿ Y).OpenCover where
  J := PUnit.{w + 1} ⊕ PUnit.{w + 1}
  obj x := x.elim (fun _ ↦ X) (fun _ ↦ Y)
  map x := x.rec (fun _ ↦ coprod.inl) (fun _ ↦ coprod.inr)
  f x := ((coprodMk X Y).symm x).elim (fun _ ↦ Sum.inl .unit) (fun _ ↦ Sum.inr .unit)
  covers x := by
    /-
      ι : Type u
      f : ι → AlgebraicGeometry.Scheme
      X Y : AlgebraicGeometry.Scheme
      x : ↑↑(CategoryTheory.Limits.coprod X Y).toPresheafedSpace
      ⊢ Membership.mem (Set.range ⇑((fun x => Sum.rec (fun x => CategoryTheory.Limit …
    -/
    obtain ⟨x, rfl⟩ := (coprodMk X Y).surjective x
    /-
      case intro
      ι : Type u
      f : ι → AlgebraicGeometry.Scheme
      X Y : AlgebraicGeometry.Scheme
      x : Sum ↑↑X.toPresheafedSpace ↑↑Y.toPresheafedSpace
      ⊢ Membership.mem (Set.range ⇑((fun x => Sum.rec (fun x => CategoryTheory.Limit …
    -/
    simp only [Sum.elim_inl, Sum.elim_inr, Set.mem_range]
    /-
      case intro
      ι : Type u
      f : ι → AlgebraicGeometry.Scheme
      X Y : AlgebraicGeometry.Scheme
      x : Sum ↑↑X.toPresheafedSpace ↑↑Y.toPresheafedSpace
      ⊢ Exists fun y => Eq ((Sum.rec (fun x => CategoryTheory.Limits.coprod.inl) (fu …
    -/
    rw [Homeomorph.symm_apply_apply]
    /-
      case intro
      ι : Type u
      f : ι → AlgebraicGeometry.Scheme
      X Y : AlgebraicGeometry.Scheme
      x : Sum ↑↑X.toPresheafedSpace ↑↑Y.toPresheafedSpace
      ⊢ Exists fun y => Eq ((Sum.rec (fun x => CategoryTheory.Limits.coprod.inl) (fu …
    -/
    obtain (x | x) := x
      /-
        case intro.inl
        ι : Type u
        f : ι → AlgebraicGeometry.Scheme
        X Y : AlgebraicGeometry.Scheme
        x : ↑↑X.toPresheafedSpace
        ⊢ Exists fun y => Eq ((Sum.rec (fun x => CategoryTheory.Limits.coprod.inl) (fu …
      -/
    · simp only [Sum.elim_inl, coprodMk_inl, exists_apply_eq_apply]
      /-
        🎉 no goals
      -/
      /-
        case intro.inr
        ι : Type u
        f : ι → AlgebraicGeometry.Scheme
        X Y : AlgebraicGeometry.Scheme
        x : ↑↑Y.toPresheafedSpace
        ⊢ Exists fun y => Eq ((Sum.rec (fun x => CategoryTheory.Limits.coprod.inl) (fu …
      -/
    · simp only [Sum.elim_inr, coprodMk_inr, exists_apply_eq_apply]
      /-
        🎉 no goals
      -/
  map_prop x := x.rec (fun _ ↦ inferInstance) (fun _ ↦ inferInstance)


/-- The map `Spec R ⨿ Spec S ⟶ Spec (R × S)`.
This is an isomorphism as witnessed by an `IsIso` instance provided below. -/
noncomputable
def coprodSpec : Spec (.of R) ⨿ Spec (.of S) ⟶ Spec (.of (R × S)) :=
  coprod.desc (Spec.map (CommRingCat.ofHom <| RingHom.fst _ _))
    (Spec.map (CommRingCat.ofHom <| RingHom.snd _ _))


@[simp, reassoc]
lemma coprodSpec_inl : coprod.inl ≫ coprodSpec R S =
    Spec.map (CommRingCat.ofHom <| RingHom.fst R S) :=
  coprod.inl_desc _ _


@[simp, reassoc]
lemma coprodSpec_inr : coprod.inr ≫ coprodSpec R S =
    Spec.map (CommRingCat.ofHom <| RingHom.snd R S) :=
  coprod.inr_desc _ _


lemma coprodSpec_coprodMk (x) :
    (coprodSpec R S).base (coprodMk _ _ x) = (PrimeSpectrum.primeSpectrumProd R S).symm x := by
  /-
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    x : Sum ↑↑(AlgebraicGeometry.Spec (CommRingCat.of R)).toPresheafedSpace ↑↑(Alg …
    ⊢ Eq ((AlgebraicGeometry.coprodSpec R S).base ((AlgebraicGeometry.coprodMk (Al …
  -/
  apply PrimeSpectrum.ext
  /-
    case asIdeal
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    x : Sum ↑↑(AlgebraicGeometry.Spec (CommRingCat.of R)).toPresheafedSpace ↑↑(Alg …
    ⊢ Eq ((AlgebraicGeometry.coprodSpec R S).base ((AlgebraicGeometry.coprodMk (Al …
  -/
  obtain (x | x) := x <;>
    simp only [coprodMk_inl, coprodMk_inr, ← Scheme.comp_base_apply,
      coprodSpec, coprod.inl_desc, coprod.inr_desc]
    /-
      case asIdeal.inl
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      x : ↑↑(AlgebraicGeometry.Spec (CommRingCat.of R)).toPresheafedSpace
      ⊢ Eq ((AlgebraicGeometry.Spec.map (CommRingCat.ofHom (RingHom.fst R S))).base  …
    -/
  · show Ideal.comap _ _ = x.asIdeal.prod ⊤
    /-
      case asIdeal.inl
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      x : ↑↑(AlgebraicGeometry.Spec (CommRingCat.of R)).toPresheafedSpace
      ⊢ Eq (Ideal.comap (CommRingCat.ofHom (RingHom.fst R S)).hom x.asIdeal) (x.asId …
    -/
    ext; simp [Ideal.prod, CommRingCat.ofHom]
         /-
           🎉 no goals
         -/
    /-
      case asIdeal.inr
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      x : ↑↑(AlgebraicGeometry.Spec (CommRingCat.of S)).toPresheafedSpace
      ⊢ Eq ((AlgebraicGeometry.Spec.map (CommRingCat.ofHom (RingHom.snd R S))).base  …
    -/
  · show Ideal.comap _ _ = Ideal.prod ⊤ x.asIdeal
    /-
      case asIdeal.inr
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      x : ↑↑(AlgebraicGeometry.Spec (CommRingCat.of S)).toPresheafedSpace
      ⊢ Eq (Ideal.comap (CommRingCat.ofHom (RingHom.snd R S)).hom x.asIdeal) (Top.to …
    -/
    ext; simp [Ideal.prod, CommRingCat.ofHom]
         /-
           🎉 no goals
         -/


lemma coprodSpec_apply (x) :
    (coprodSpec R S).base x = (PrimeSpectrum.primeSpectrumProd R S).symm
      ((coprodMk (Spec (.of R)) (Spec (.of S))).symm x) := by
  /-
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    x : ↑↑(CategoryTheory.Limits.coprod (AlgebraicGeometry.Spec (CommRingCat.of R) …
    ⊢ Eq ((AlgebraicGeometry.coprodSpec R S).base x) ((PrimeSpectrum.primeSpectrum …
  -/
  rw [← coprodSpec_coprodMk, Homeomorph.apply_symm_apply]
  /-
    🎉 no goals
  -/


lemma isIso_stalkMap_coprodSpec (x) :
    IsIso ((coprodSpec R S).stalkMap x) := by
  /-
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    x : ↑↑(CategoryTheory.Limits.coprod (AlgebraicGeometry.Spec (CommRingCat.of R) …
    ⊢ CategoryTheory.IsIso (AlgebraicGeometry.Scheme.Hom.stalkMap (AlgebraicGeomet …
  -/
  obtain ⟨x | x, rfl⟩ := (coprodMk _ _).surjective x
    /-
      case intro.inl
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      x : ↑↑(AlgebraicGeometry.Spec (CommRingCat.of R)).toPresheafedSpace
      ⊢ CategoryTheory.IsIso (AlgebraicGeometry.Scheme.Hom.stalkMap (AlgebraicGeomet …
    -/
  · have := Scheme.stalkMap_comp coprod.inl (coprodSpec R S) x
    /-
      case intro.inl
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      x : ↑↑(AlgebraicGeometry.Spec (CommRingCat.of R)).toPresheafedSpace
      this : Eq (AlgebraicGeometry.Scheme.Hom.stalkMap (CategoryTheory.CategoryStruc …
      ⊢ CategoryTheory.IsIso (AlgebraicGeometry.Scheme.Hom.stalkMap (AlgebraicGeomet …
    -/
    rw [← IsIso.comp_inv_eq, Scheme.stalkMap_congr_hom _ (Spec.map _) (coprodSpec_inl R S)] at this
    /-
      case intro.inl
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      x : ↑↑(AlgebraicGeometry.Spec (CommRingCat.of R)).toPresheafedSpace
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
      ⊢ CategoryTheory.IsIso (AlgebraicGeometry.Scheme.Hom.stalkMap (AlgebraicGeomet …
    -/
    rw [coprodMk_inl, ← this]
    /-
      case intro.inl
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      x : ↑↑(AlgebraicGeometry.Spec (CommRingCat.of R)).toPresheafedSpace
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
      ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheory.Cat …
    -/
    letI := (RingHom.fst R S).toAlgebra
    /-
      case intro.inl
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      x : ↑↑(AlgebraicGeometry.Spec (CommRingCat.of R)).toPresheafedSpace
      this✝ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct. …
      this : Algebra (Prod R S) R := (RingHom.fst R S).toAlgebra
      ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheory.Cat …
    -/
    have := IsLocalization.away_fst (R := R) (S := S)
    have : IsOpenImmersion (Spec.map (CommRingCat.ofHom (RingHom.fst R S))) :=
      IsOpenImmersion.of_isLocalization (1, 0)
    /-
      case intro.inl
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      x : ↑↑(AlgebraicGeometry.Spec (CommRingCat.of R)).toPresheafedSpace
      this✝² : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct …
      this✝¹ : Algebra (Prod R S) R := (RingHom.fst R S).toAlgebra
      this✝ : IsLocalization.Away { fst := 1, snd := 0 } R
      this : AlgebraicGeometry.IsOpenImmersion (AlgebraicGeometry.Spec.map (CommRing …
      ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheory.Cat …
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      x : ↑↑(AlgebraicGeometry.Spec (CommRingCat.of S)).toPresheafedSpace
      ⊢ CategoryTheory.IsIso (AlgebraicGeometry.Scheme.Hom.stalkMap (AlgebraicGeomet …
    -/
  · have := Scheme.stalkMap_comp coprod.inr (coprodSpec R S) x
    /-
      case intro.inr
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      x : ↑↑(AlgebraicGeometry.Spec (CommRingCat.of S)).toPresheafedSpace
      this : Eq (AlgebraicGeometry.Scheme.Hom.stalkMap (CategoryTheory.CategoryStruc …
      ⊢ CategoryTheory.IsIso (AlgebraicGeometry.Scheme.Hom.stalkMap (AlgebraicGeomet …
    -/
    rw [← IsIso.comp_inv_eq, Scheme.stalkMap_congr_hom _ (Spec.map _) (coprodSpec_inr R S)] at this
    /-
      case intro.inr
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      x : ↑↑(AlgebraicGeometry.Spec (CommRingCat.of S)).toPresheafedSpace
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
      ⊢ CategoryTheory.IsIso (AlgebraicGeometry.Scheme.Hom.stalkMap (AlgebraicGeomet …
    -/
    rw [coprodMk_inr, ← this]
    /-
      case intro.inr
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      x : ↑↑(AlgebraicGeometry.Spec (CommRingCat.of S)).toPresheafedSpace
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
      ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheory.Cat …
    -/
    letI := (RingHom.snd R S).toAlgebra
    /-
      case intro.inr
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      x : ↑↑(AlgebraicGeometry.Spec (CommRingCat.of S)).toPresheafedSpace
      this✝ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct. …
      this : Algebra (Prod R S) S := (RingHom.snd R S).toAlgebra
      ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheory.Cat …
    -/
    have := IsLocalization.away_snd (R := R) (S := S)
    have : IsOpenImmersion (Spec.map (CommRingCat.ofHom (RingHom.snd R S))) :=
      IsOpenImmersion.of_isLocalization (0, 1)
    /-
      case intro.inr
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      x : ↑↑(AlgebraicGeometry.Spec (CommRingCat.of S)).toPresheafedSpace
      this✝² : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct …
      this✝¹ : Algebra (Prod R S) S := (RingHom.snd R S).toAlgebra
      this✝ : IsLocalization.Away { fst := 0, snd := 1 } S
      this : AlgebraicGeometry.IsOpenImmersion (AlgebraicGeometry.Spec.map (CommRing …
      ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheory.Cat …
    -/
    infer_instance
    /-
      🎉 no goals
    -/


instance : IsIso (coprodSpec R S) := by
  /-
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    X Y : AlgebraicGeometry.Scheme
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    ⊢ CategoryTheory.IsIso (AlgebraicGeometry.coprodSpec R S)
  -/
  rw [isIso_iff_stalk_iso]
  /-
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    X Y : AlgebraicGeometry.Scheme
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    ⊢ And (CategoryTheory.IsIso (AlgebraicGeometry.coprodSpec R S).base) (∀ (x : ↑ …
  -/
  refine ⟨?_, isIso_stalkMap_coprodSpec R S⟩
  convert_to IsIso (TopCat.isoOfHomeo (X := Spec (.of (R × S))) <|
    PrimeSpectrum.primeSpectrumProdHomeo.trans (coprodMk (Spec (.of R)) (Spec (.of S)))).inv
    /-
      case h.e'_5
      ι : Type u
      f : ι → AlgebraicGeometry.Scheme
      X Y : AlgebraicGeometry.Scheme
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      ⊢ Eq (AlgebraicGeometry.coprodSpec R S).base (TopCat.isoOfHomeo (PrimeSpectrum …
    -/
  · ext x; exact coprodSpec_apply R S x
           /-
             🎉 no goals
           -/
    /-
      ι : Type u
      f : ι → AlgebraicGeometry.Scheme
      X Y : AlgebraicGeometry.Scheme
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      ⊢ CategoryTheory.IsIso (TopCat.isoOfHomeo (PrimeSpectrum.primeSpectrumProdHome …
    -/
  · infer_instance
    /-
      🎉 no goals
    -/


instance (R S : CommRingCatᵒᵖ) : IsIso (coprodComparison Scheme.Spec R S) := by
  /-
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    X Y : AlgebraicGeometry.Scheme
    R✝ S✝ : Type u
    inst✝¹ : CommRing R✝
    inst✝ : CommRing S✝
    R S : Opposite CommRingCat
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.coprodComparison AlgebraicGeomet …
  -/
  obtain ⟨R⟩ := R; obtain ⟨S⟩ := S
  have : coprodComparison Scheme.Spec (.op R) (.op S) ≫ (Spec.map
    ((limit.isoLimitCone ⟨_, CommRingCat.prodFanIsLimit R S⟩).inv ≫
      (opProdIsoCoprod R S).unop.inv)) = coprodSpec R S := by
    ext1
    · rw [coprodComparison_inl_assoc, coprodSpec, coprod.inl_desc, Scheme.Spec_map,
        ← Spec.map_comp, Category.assoc, Iso.unop_inv, opProdIsoCoprod_inv_inl,
        limit.isoLimitCone_inv_π]
      rfl
    · rw [coprodComparison_inr_assoc, coprodSpec, coprod.inr_desc, Scheme.Spec_map,
        ← Spec.map_comp, Category.assoc, Iso.unop_inv, opProdIsoCoprod_inv_inr,
        limit.isoLimitCone_inv_π]
      rfl
  /-
    case op.op
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    X Y : AlgebraicGeometry.Scheme
    R✝ S✝ : Type u
    inst✝¹ : CommRing R✝
    inst✝ : CommRing S✝
    R S : CommRingCat
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coprodCom …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.coprodComparison AlgebraicGeomet …
  -/
  rw [(IsIso.eq_comp_inv _).mpr this]
  /-
    case op.op
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    X Y : AlgebraicGeometry.Scheme
    R✝ S✝ : Type u
    inst✝¹ : CommRing R✝
    inst✝ : CommRing S✝
    R S : CommRingCat
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coprodCom …
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry. …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


noncomputable
instance : PreservesColimitsOfShape (Discrete WalkingPair) Scheme.Spec :=
  ⟨fun {_} ↦
    have (X Y : CommRingCatᵒᵖ) := PreservesColimitPair.of_iso_coprod_comparison Scheme.Spec X Y
    preservesColimit_of_iso_diagram _ (diagramIsoPair _).symm⟩


noncomputable
instance : PreservesColimitsOfShape (Discrete PEmpty.{1}) Scheme.Spec := by
  have : IsEmpty (Scheme.Spec.obj (⊥_ CommRingCatᵒᵖ)) :=
    @Function.isEmpty _ _ spec_punit_isEmpty (Scheme.Spec.mapIso
      (initialIsoIsInitial (initialOpOfTerminal CommRingCat.punitIsTerminal))).hom.base
  /-
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    X Y : AlgebraicGeometry.Scheme
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    this : IsEmpty ↑↑(AlgebraicGeometry.Scheme.Spec.obj (CategoryTheory.Limits.ini …
    ⊢ CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discrete PEmp …
  -/
  have := preservesInitial_of_iso Scheme.Spec (asIso (initial.to _))
  /-
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    X Y : AlgebraicGeometry.Scheme
    R S : Type u
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    this✝ : IsEmpty ↑↑(AlgebraicGeometry.Scheme.Spec.obj (CategoryTheory.Limits.in …
    this : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Functor.empty (O …
    ⊢ CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discrete PEmp …
  -/
  exact preservesColimitsOfShape_pempty_of_preservesInitial _
  /-
    🎉 no goals
  -/


noncomputable
instance {J} [Fintype J] : PreservesColimitsOfShape (Discrete J) Scheme.Spec :=
  preservesFiniteCoproductsOfPreservesBinaryAndInitial _ _


noncomputable
instance {J : Type*} [Finite J] : PreservesColimitsOfShape (Discrete J) Scheme.Spec :=
  letI := (nonempty_fintype J).some
  preservesColimitsOfShape_of_equiv (Discrete.equivalence (Fintype.equivFin _).symm) _


/-- The canonical map `∐ Spec Rᵢ ⟶ Spec (Π Rᵢ)`.
This is an isomorphism when the product is finite. -/
noncomputable
def sigmaSpec (R : ι → CommRingCat) : (∐ fun i ↦ Spec (R i)) ⟶ Spec (.of (Π i, R i)) :=
  Sigma.desc (fun i ↦ Spec.map (CommRingCat.ofHom (Pi.evalRingHom _ i)))


@[simp, reassoc]
lemma ι_sigmaSpec (R : ι → CommRingCat) (i) :
    Sigma.ι _ i ≫ sigmaSpec R = Spec.map (CommRingCat.ofHom (Pi.evalRingHom _ i)) :=
  Sigma.ι_desc _ _


instance [Finite ι] (R : ι → CommRingCat) : IsIso (sigmaSpec R) := by
  have : sigmaSpec R =
      (colimit.isoColimitCocone ⟨_,
        (IsColimit.precomposeHomEquiv Discrete.natIsoFunctor.symm _).symm (isColimitOfPreserves
          Scheme.Spec (Fan.IsLimit.op (CommRingCat.piFanIsLimit R)))⟩).hom := by
    ext1; simp; rfl
  /-
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    X Y : AlgebraicGeometry.Scheme
    R✝ S : Type u
    inst✝² : CommRing R✝
    inst✝¹ : CommRing S
    inst✝ : Finite ι
    R : ι → CommRingCat
    this : Eq (AlgebraicGeometry.sigmaSpec R) (CategoryTheory.Limits.colimit.isoCo …
    ⊢ CategoryTheory.IsIso (AlgebraicGeometry.sigmaSpec R)
  -/
  rw [this]
  /-
    ι : Type u
    f : ι → AlgebraicGeometry.Scheme
    X Y : AlgebraicGeometry.Scheme
    R✝ S : Type u
    inst✝² : CommRing R✝
    inst✝¹ : CommRing S
    inst✝ : Finite ι
    R : ι → CommRingCat
    this : Eq (AlgebraicGeometry.sigmaSpec R) (CategoryTheory.Limits.colimit.isoCo …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.colimit.isoColimitCocone { cocon …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance [Finite ι] [∀ i, IsAffine (f i)] : IsAffine (∐ f) :=
  isAffine_of_isIso ((Sigma.mapIso (fun i ↦ (f i).isoSpec)).hom ≫ sigmaSpec _)


instance [IsAffine X] [IsAffine Y] : IsAffine (X ⨿ Y) :=
  isAffine_of_isIso ((coprod.mapIso X.isoSpec Y.isoSpec).hom ≫ coprodSpec _ _)


