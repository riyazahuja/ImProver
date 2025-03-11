/-- A morphism of homological complexes `f : K ⟶ L` is a quasi-isomorphism in degree `i`
when it induces a quasi-isomorphism of short complexes `K.sc i ⟶ L.sc i`. -/
class QuasiIsoAt (f : K ⟶ L) (i : ι) [K.HasHomology i] [L.HasHomology i] : Prop where
  quasiIso : ShortComplex.QuasiIso ((shortComplexFunctor C c i).map f)


lemma quasiIsoAt_iff (f : K ⟶ L) (i : ι) [K.HasHomology i] [L.HasHomology i] :
    QuasiIsoAt f i ↔
      ShortComplex.QuasiIso ((shortComplexFunctor C c i).map f) := by
  /-
    ι : Type u_1
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K L : HomologicalComplex C c
    f : Quiver.Hom K L
    i : ι
    inst✝¹ : K.HasHomology i
    inst✝ : L.HasHomology i
    ⊢ Iff (QuasiIsoAt f i) (CategoryTheory.ShortComplex.QuasiIso ((HomologicalComp …
  -/
  constructor
    /-
      case mp
      ι : Type u_1
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      c : ComplexShape ι
      K L : HomologicalComplex C c
      f : Quiver.Hom K L
      i : ι
      inst✝¹ : K.HasHomology i
      inst✝ : L.HasHomology i
      ⊢ QuasiIsoAt f i → CategoryTheory.ShortComplex.QuasiIso ((HomologicalComplex.s …
    -/
  · intro h
    /-
      case mp
      ι : Type u_1
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      c : ComplexShape ι
      K L : HomologicalComplex C c
      f : Quiver.Hom K L
      i : ι
      inst✝¹ : K.HasHomology i
      inst✝ : L.HasHomology i
      h : QuasiIsoAt f i
      ⊢ CategoryTheory.ShortComplex.QuasiIso ((HomologicalComplex.shortComplexFuncto …
    -/
    exact h.quasiIso
    /-
      🎉 no goals
    -/
    /-
      case mpr
      ι : Type u_1
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      c : ComplexShape ι
      K L : HomologicalComplex C c
      f : Quiver.Hom K L
      i : ι
      inst✝¹ : K.HasHomology i
      inst✝ : L.HasHomology i
      ⊢ CategoryTheory.ShortComplex.QuasiIso ((HomologicalComplex.shortComplexFuncto …
    -/
  · intro h
    /-
      case mpr
      ι : Type u_1
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      c : ComplexShape ι
      K L : HomologicalComplex C c
      f : Quiver.Hom K L
      i : ι
      inst✝¹ : K.HasHomology i
      inst✝ : L.HasHomology i
      h : CategoryTheory.ShortComplex.QuasiIso ((HomologicalComplex.shortComplexFunc …
      ⊢ QuasiIsoAt f i
    -/
    exact ⟨h⟩
    /-
      🎉 no goals
    -/


instance quasiIsoAt_of_isIso (f : K ⟶ L) [IsIso f] (i : ι) [K.HasHomology i] [L.HasHomology i] :
    QuasiIsoAt f i := by
  /-
    ι : Type u_1
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K L M K' L' : HomologicalComplex C c
    f : Quiver.Hom K L
    inst✝² : CategoryTheory.IsIso f
    i : ι
    inst✝¹ : K.HasHomology i
    inst✝ : L.HasHomology i
    ⊢ QuasiIsoAt f i
  -/
  rw [quasiIsoAt_iff]
  /-
    ι : Type u_1
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K L M K' L' : HomologicalComplex C c
    f : Quiver.Hom K L
    inst✝² : CategoryTheory.IsIso f
    i : ι
    inst✝¹ : K.HasHomology i
    inst✝ : L.HasHomology i
    ⊢ CategoryTheory.ShortComplex.QuasiIso ((HomologicalComplex.shortComplexFuncto …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma quasiIsoAt_iff' (f : K ⟶ L) (i j k : ι) (hi : c.prev j = i) (hk : c.next j = k)
    [K.HasHomology j] [L.HasHomology j] [(K.sc' i j k).HasHomology] [(L.sc' i j k).HasHomology] :
    QuasiIsoAt f j ↔
      ShortComplex.QuasiIso ((shortComplexFunctor' C c i j k).map f) := by
  /-
    ι : Type u_1
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K L : HomologicalComplex C c
    f : Quiver.Hom K L
    i j k : ι
    hi : Eq (c.prev j) i
    hk : Eq (c.next j) k
    inst✝³ : K.HasHomology j
    inst✝² : L.HasHomology j
    inst✝¹ : (K.sc' i j k).HasHomology
    inst✝ : (L.sc' i j k).HasHomology
    ⊢ Iff (QuasiIsoAt f j) (CategoryTheory.ShortComplex.QuasiIso ((HomologicalComp …
  -/
  rw [quasiIsoAt_iff]
  exact ShortComplex.quasiIso_iff_of_arrow_mk_iso _ _
    (Arrow.isoOfNatIso (natIsoSc' C c i j k hi hk) (Arrow.mk f))


lemma quasiIsoAt_iff_isIso_homologyMap (f : K ⟶ L) (i : ι)
    [K.HasHomology i] [L.HasHomology i] :
    QuasiIsoAt f i ↔ IsIso (homologyMap f i) := by
  /-
    ι : Type u_1
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K L : HomologicalComplex C c
    f : Quiver.Hom K L
    i : ι
    inst✝¹ : K.HasHomology i
    inst✝ : L.HasHomology i
    ⊢ Iff (QuasiIsoAt f i) (CategoryTheory.IsIso (HomologicalComplex.homologyMap f …
  -/
  rw [quasiIsoAt_iff, ShortComplex.quasiIso_iff]
  /-
    ι : Type u_1
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K L : HomologicalComplex C c
    f : Quiver.Hom K L
    i : ι
    inst✝¹ : K.HasHomology i
    inst✝ : L.HasHomology i
    ⊢ Iff (CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap ((Homolog …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma quasiIsoAt_iff_exactAt (f : K ⟶ L) (i : ι) [K.HasHomology i] [L.HasHomology i]
    (hK : K.ExactAt i) :
    QuasiIsoAt f i ↔ L.ExactAt i := by
  simp only [quasiIsoAt_iff, ShortComplex.quasiIso_iff, exactAt_iff,
    ShortComplex.exact_iff_isZero_homology] at hK ⊢
  /-
    ι : Type u_1
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K L : HomologicalComplex C c
    f : Quiver.Hom K L
    i : ι
    inst✝¹ : K.HasHomology i
    inst✝ : L.HasHomology i
    hK : CategoryTheory.Limits.IsZero (K.sc i).homology
    ⊢ Iff (CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap ((Homolog …
  -/
  constructor
    /-
      case mp
      ι : Type u_1
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      c : ComplexShape ι
      K L : HomologicalComplex C c
      f : Quiver.Hom K L
      i : ι
      inst✝¹ : K.HasHomology i
      inst✝ : L.HasHomology i
      hK : CategoryTheory.Limits.IsZero (K.sc i).homology
      ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap ((HomologicalC …
    -/
  · intro h
    /-
      case mp
      ι : Type u_1
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      c : ComplexShape ι
      K L : HomologicalComplex C c
      f : Quiver.Hom K L
      i : ι
      inst✝¹ : K.HasHomology i
      inst✝ : L.HasHomology i
      hK : CategoryTheory.Limits.IsZero (K.sc i).homology
      h : CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap ((Homologica …
      ⊢ CategoryTheory.Limits.IsZero (L.sc i).homology
    -/
    exact IsZero.of_iso hK (@asIso _ _ _ _ _ h).symm
    /-
      🎉 no goals
    -/
    /-
      case mpr
      ι : Type u_1
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      c : ComplexShape ι
      K L : HomologicalComplex C c
      f : Quiver.Hom K L
      i : ι
      inst✝¹ : K.HasHomology i
      inst✝ : L.HasHomology i
      hK : CategoryTheory.Limits.IsZero (K.sc i).homology
      ⊢ CategoryTheory.Limits.IsZero (L.sc i).homology → CategoryTheory.IsIso (Categ …
    -/
  · intro hL
    /-
      case mpr
      ι : Type u_1
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      c : ComplexShape ι
      K L : HomologicalComplex C c
      f : Quiver.Hom K L
      i : ι
      inst✝¹ : K.HasHomology i
      inst✝ : L.HasHomology i
      hK : CategoryTheory.Limits.IsZero (K.sc i).homology
      hL : CategoryTheory.Limits.IsZero (L.sc i).homology
      ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap ((HomologicalC …
    -/
    exact ⟨⟨0, IsZero.eq_of_src hK _ _, IsZero.eq_of_tgt hL _ _⟩⟩
    /-
      🎉 no goals
    -/


lemma quasiIsoAt_iff_exactAt' (f : K ⟶ L) (i : ι) [K.HasHomology i] [L.HasHomology i]
    (hL : L.ExactAt i) :
    QuasiIsoAt f i ↔ K.ExactAt i := by
  simp only [quasiIsoAt_iff, ShortComplex.quasiIso_iff, exactAt_iff,
    ShortComplex.exact_iff_isZero_homology] at hL ⊢
  /-
    ι : Type u_1
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K L : HomologicalComplex C c
    f : Quiver.Hom K L
    i : ι
    inst✝¹ : K.HasHomology i
    inst✝ : L.HasHomology i
    hL : CategoryTheory.Limits.IsZero (L.sc i).homology
    ⊢ Iff (CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap ((Homolog …
  -/
  constructor
    /-
      case mp
      ι : Type u_1
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      c : ComplexShape ι
      K L : HomologicalComplex C c
      f : Quiver.Hom K L
      i : ι
      inst✝¹ : K.HasHomology i
      inst✝ : L.HasHomology i
      hL : CategoryTheory.Limits.IsZero (L.sc i).homology
      ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap ((HomologicalC …
    -/
  · intro h
    /-
      case mp
      ι : Type u_1
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      c : ComplexShape ι
      K L : HomologicalComplex C c
      f : Quiver.Hom K L
      i : ι
      inst✝¹ : K.HasHomology i
      inst✝ : L.HasHomology i
      hL : CategoryTheory.Limits.IsZero (L.sc i).homology
      h : CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap ((Homologica …
      ⊢ CategoryTheory.Limits.IsZero (K.sc i).homology
    -/
    exact IsZero.of_iso hL (@asIso _ _ _ _ _ h)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      ι : Type u_1
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      c : ComplexShape ι
      K L : HomologicalComplex C c
      f : Quiver.Hom K L
      i : ι
      inst✝¹ : K.HasHomology i
      inst✝ : L.HasHomology i
      hL : CategoryTheory.Limits.IsZero (L.sc i).homology
      ⊢ CategoryTheory.Limits.IsZero (K.sc i).homology → CategoryTheory.IsIso (Categ …
    -/
  · intro hK
    /-
      case mpr
      ι : Type u_1
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      c : ComplexShape ι
      K L : HomologicalComplex C c
      f : Quiver.Hom K L
      i : ι
      inst✝¹ : K.HasHomology i
      inst✝ : L.HasHomology i
      hL : CategoryTheory.Limits.IsZero (L.sc i).homology
      hK : CategoryTheory.Limits.IsZero (K.sc i).homology
      ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap ((HomologicalC …
    -/
    exact ⟨⟨0, IsZero.eq_of_src hK _ _, IsZero.eq_of_tgt hL _ _⟩⟩
    /-
      🎉 no goals
    -/


instance (f : K ⟶ L) (i : ι) [K.HasHomology i] [L.HasHomology i] [hf : QuasiIsoAt f i] :
    IsIso (homologyMap f i) := by
  /-
    ι : Type u_1
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K L M K' L' : HomologicalComplex C c
    f : Quiver.Hom K L
    i : ι
    inst✝¹ : K.HasHomology i
    inst✝ : L.HasHomology i
    hf : QuasiIsoAt f i
    ⊢ CategoryTheory.IsIso (HomologicalComplex.homologyMap f i)
  -/
  simpa only [quasiIsoAt_iff, ShortComplex.quasiIso_iff] using hf
  /-
    🎉 no goals
  -/


/-- The isomorphism `K.homology i ≅ L.homology i` induced by a morphism `f : K ⟶ L` such
that `[QuasiIsoAt f i]` holds. -/
@[simps! hom]
noncomputable def isoOfQuasiIsoAt (f : K ⟶ L) (i : ι) [K.HasHomology i] [L.HasHomology i]
    [QuasiIsoAt f i] : K.homology i ≅ L.homology i :=
  asIso (homologyMap f i)


@[reassoc (attr := simp)]
lemma isoOfQuasiIsoAt_hom_inv_id (f : K ⟶ L) (i : ι) [K.HasHomology i] [L.HasHomology i]
    [QuasiIsoAt f i] :
    homologyMap f i ≫ (isoOfQuasiIsoAt f i).inv = 𝟙 _ :=
  (isoOfQuasiIsoAt f i).hom_inv_id


@[reassoc (attr := simp)]
lemma isoOfQuasiIsoAt_inv_hom_id (f : K ⟶ L) (i : ι) [K.HasHomology i] [L.HasHomology i]
    [QuasiIsoAt f i] :
    (isoOfQuasiIsoAt f i).inv ≫ homologyMap f i = 𝟙 _ :=
  (isoOfQuasiIsoAt f i).inv_hom_id


lemma CochainComplex.quasiIsoAt₀_iff {K L : CochainComplex C ℕ} (f : K ⟶ L)
    [K.HasHomology 0] [L.HasHomology 0] [(K.sc' 0 0 1).HasHomology] [(L.sc' 0 0 1).HasHomology] :
    QuasiIsoAt f 0 ↔
      ShortComplex.QuasiIso ((HomologicalComplex.shortComplexFunctor' C _ 0 0 1).map f) :=
                              /-
                                C : Type u
                                inst✝⁵ : CategoryTheory.Category.{v, u} C
                                inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
                                K L : CochainComplex C Nat
                                f : Quiver.Hom K L
                                inst✝³ : HomologicalComplex.HasHomology K 0
                                inst✝² : HomologicalComplex.HasHomology L 0
                                inst✝¹ : (HomologicalComplex.sc' K 0 0 1).HasHomology
                                inst✝ : (HomologicalComplex.sc' L 0 0 1).HasHomology
                                ⊢ Eq ((ComplexShape.up Nat).prev 0) 0
                              -/
                              /-
                                🎉 no goals
                              -/
  quasiIsoAt_iff' _ _ _ _ (by simp) (by simp)
                                        /-
                                          🎉 no goals
                                        -/


lemma ChainComplex.quasiIsoAt₀_iff {K L : ChainComplex C ℕ} (f : K ⟶ L)
    [K.HasHomology 0] [L.HasHomology 0] [(K.sc' 1 0 0).HasHomology] [(L.sc' 1 0 0).HasHomology] :
    QuasiIsoAt f 0 ↔
      ShortComplex.QuasiIso ((HomologicalComplex.shortComplexFunctor' C _ 1 0 0).map f) :=
                              /-
                                C : Type u
                                inst✝⁵ : CategoryTheory.Category.{v, u} C
                                inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
                                K L : ChainComplex C Nat
                                f : Quiver.Hom K L
                                inst✝³ : HomologicalComplex.HasHomology K 0
                                inst✝² : HomologicalComplex.HasHomology L 0
                                inst✝¹ : (HomologicalComplex.sc' K 1 0 0).HasHomology
                                inst✝ : (HomologicalComplex.sc' L 1 0 0).HasHomology
                                ⊢ Eq ((ComplexShape.down Nat).prev 0) 1
                              -/
                              /-
                                🎉 no goals
                              -/
  quasiIsoAt_iff' _ _ _ _ (by simp) (by simp)
                                        /-
                                          🎉 no goals
                                        -/


/-- A morphism of homological complexes `f : K ⟶ L` is a quasi-isomorphism when it
is so in every degree, i.e. when the induced maps `homologyMap f i : K.homology i ⟶ L.homology i`
are all isomorphisms (see `quasiIso_iff` and `quasiIsoAt_iff_isIso_homologyMap`). -/
class QuasiIso (f : K ⟶ L) [∀ i, K.HasHomology i] [∀ i, L.HasHomology i] : Prop where
  quasiIsoAt : ∀ i, QuasiIsoAt f i := by infer_instance


lemma quasiIso_iff (f : K ⟶ L) [∀ i, K.HasHomology i] [∀ i, L.HasHomology i] :
    QuasiIso f ↔ ∀ i, QuasiIsoAt f i :=
  ⟨fun h => h.quasiIsoAt, fun h => ⟨h⟩⟩


instance quasiIso_of_isIso (f : K ⟶ L) [IsIso f] [∀ i, K.HasHomology i] [∀ i, L.HasHomology i] :
    QuasiIso f where


instance quasiIsoAt_comp (φ : K ⟶ L) (φ' : L ⟶ M) (i : ι) [K.HasHomology i]
    [L.HasHomology i] [M.HasHomology i]
    [hφ : QuasiIsoAt φ i] [hφ' : QuasiIsoAt φ' i] :
    QuasiIsoAt (φ ≫ φ') i := by
  /-
    ι : Type u_1
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K L M K' L' : HomologicalComplex C c
    φ : Quiver.Hom K L
    φ' : Quiver.Hom L M
    i : ι
    inst✝² : K.HasHomology i
    inst✝¹ : L.HasHomology i
    inst✝ : M.HasHomology i
    hφ : QuasiIsoAt φ i
    hφ' : QuasiIsoAt φ' i
    ⊢ QuasiIsoAt (CategoryTheory.CategoryStruct.comp φ φ') i
  -/
  rw [quasiIsoAt_iff] at hφ hφ' ⊢
  /-
    ι : Type u_1
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K L M K' L' : HomologicalComplex C c
    φ : Quiver.Hom K L
    φ' : Quiver.Hom L M
    i : ι
    inst✝² : K.HasHomology i
    inst✝¹ : L.HasHomology i
    inst✝ : M.HasHomology i
    hφ : CategoryTheory.ShortComplex.QuasiIso ((HomologicalComplex.shortComplexFun …
    hφ' : CategoryTheory.ShortComplex.QuasiIso ((HomologicalComplex.shortComplexFu …
    ⊢ CategoryTheory.ShortComplex.QuasiIso ((HomologicalComplex.shortComplexFuncto …
  -/
  rw [Functor.map_comp]
  /-
    ι : Type u_1
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K L M K' L' : HomologicalComplex C c
    φ : Quiver.Hom K L
    φ' : Quiver.Hom L M
    i : ι
    inst✝² : K.HasHomology i
    inst✝¹ : L.HasHomology i
    inst✝ : M.HasHomology i
    hφ : CategoryTheory.ShortComplex.QuasiIso ((HomologicalComplex.shortComplexFun …
    hφ' : CategoryTheory.ShortComplex.QuasiIso ((HomologicalComplex.shortComplexFu …
    ⊢ CategoryTheory.ShortComplex.QuasiIso (CategoryTheory.CategoryStruct.comp ((H …
  -/
  exact ShortComplex.quasiIso_comp _ _
  /-
    🎉 no goals
  -/


instance quasiIso_comp (φ : K ⟶ L) (φ' : L ⟶ M) [∀ i, K.HasHomology i]
    [∀ i, L.HasHomology i] [∀ i, M.HasHomology i]
    [hφ : QuasiIso φ] [hφ' : QuasiIso φ'] :
    QuasiIso (φ ≫ φ') where


lemma quasiIsoAt_of_comp_left (φ : K ⟶ L) (φ' : L ⟶ M) (i : ι) [K.HasHomology i]
    [L.HasHomology i] [M.HasHomology i]
    [hφ : QuasiIsoAt φ i] [hφφ' : QuasiIsoAt (φ ≫ φ') i] :
    QuasiIsoAt φ' i := by
  /-
    ι : Type u_1
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K L M : HomologicalComplex C c
    φ : Quiver.Hom K L
    φ' : Quiver.Hom L M
    i : ι
    inst✝² : K.HasHomology i
    inst✝¹ : L.HasHomology i
    inst✝ : M.HasHomology i
    hφ : QuasiIsoAt φ i
    hφφ' : QuasiIsoAt (CategoryTheory.CategoryStruct.comp φ φ') i
    ⊢ QuasiIsoAt φ' i
  -/
  rw [quasiIsoAt_iff_isIso_homologyMap] at hφ hφφ' ⊢
  /-
    ι : Type u_1
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K L M : HomologicalComplex C c
    φ : Quiver.Hom K L
    φ' : Quiver.Hom L M
    i : ι
    inst✝² : K.HasHomology i
    inst✝¹ : L.HasHomology i
    inst✝ : M.HasHomology i
    hφ : CategoryTheory.IsIso (HomologicalComplex.homologyMap φ i)
    hφφ' : CategoryTheory.IsIso (HomologicalComplex.homologyMap (CategoryTheory.Ca …
    ⊢ CategoryTheory.IsIso (HomologicalComplex.homologyMap φ' i)
  -/
  rw [homologyMap_comp] at hφφ'
  /-
    ι : Type u_1
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K L M : HomologicalComplex C c
    φ : Quiver.Hom K L
    φ' : Quiver.Hom L M
    i : ι
    inst✝² : K.HasHomology i
    inst✝¹ : L.HasHomology i
    inst✝ : M.HasHomology i
    hφ : CategoryTheory.IsIso (HomologicalComplex.homologyMap φ i)
    hφφ' : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (HomologicalCo …
    ⊢ CategoryTheory.IsIso (HomologicalComplex.homologyMap φ' i)
  -/
  exact IsIso.of_isIso_comp_left (homologyMap φ i) (homologyMap φ' i)
  /-
    🎉 no goals
  -/


lemma quasiIsoAt_iff_comp_left (φ : K ⟶ L) (φ' : L ⟶ M) (i : ι) [K.HasHomology i]
    [L.HasHomology i] [M.HasHomology i]
    [hφ : QuasiIsoAt φ i] :
    QuasiIsoAt (φ ≫ φ') i ↔ QuasiIsoAt φ' i := by
  /-
    ι : Type u_1
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K L M : HomologicalComplex C c
    φ : Quiver.Hom K L
    φ' : Quiver.Hom L M
    i : ι
    inst✝² : K.HasHomology i
    inst✝¹ : L.HasHomology i
    inst✝ : M.HasHomology i
    hφ : QuasiIsoAt φ i
    ⊢ Iff (QuasiIsoAt (CategoryTheory.CategoryStruct.comp φ φ') i) (QuasiIsoAt φ' i)
  -/
  constructor
    /-
      case mp
      ι : Type u_1
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      c : ComplexShape ι
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      i : ι
      inst✝² : K.HasHomology i
      inst✝¹ : L.HasHomology i
      inst✝ : M.HasHomology i
      hφ : QuasiIsoAt φ i
      ⊢ QuasiIsoAt (CategoryTheory.CategoryStruct.comp φ φ') i → QuasiIsoAt φ' i
    -/
  · intro
    /-
      case mp
      ι : Type u_1
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      c : ComplexShape ι
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      i : ι
      inst✝² : K.HasHomology i
      inst✝¹ : L.HasHomology i
      inst✝ : M.HasHomology i
      hφ : QuasiIsoAt φ i
      a✝ : QuasiIsoAt (CategoryTheory.CategoryStruct.comp φ φ') i
      ⊢ QuasiIsoAt φ' i
    -/
    exact quasiIsoAt_of_comp_left φ φ' i
    /-
      🎉 no goals
    -/
    /-
      case mpr
      ι : Type u_1
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      c : ComplexShape ι
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      i : ι
      inst✝² : K.HasHomology i
      inst✝¹ : L.HasHomology i
      inst✝ : M.HasHomology i
      hφ : QuasiIsoAt φ i
      ⊢ QuasiIsoAt φ' i → QuasiIsoAt (CategoryTheory.CategoryStruct.comp φ φ') i
    -/
  · intro
    /-
      case mpr
      ι : Type u_1
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      c : ComplexShape ι
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      i : ι
      inst✝² : K.HasHomology i
      inst✝¹ : L.HasHomology i
      inst✝ : M.HasHomology i
      hφ : QuasiIsoAt φ i
      a✝ : QuasiIsoAt φ' i
      ⊢ QuasiIsoAt (CategoryTheory.CategoryStruct.comp φ φ') i
    -/
    infer_instance
    /-
      🎉 no goals
    -/


lemma quasiIso_iff_comp_left (φ : K ⟶ L) (φ' : L ⟶ M) [∀ i, K.HasHomology i]
    [∀ i, L.HasHomology i] [∀ i, M.HasHomology i]
    [hφ : QuasiIso φ] :
    QuasiIso (φ ≫ φ') ↔ QuasiIso φ' := by
  /-
    ι : Type u_1
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K L M : HomologicalComplex C c
    φ : Quiver.Hom K L
    φ' : Quiver.Hom L M
    inst✝² : ∀ (i : ι), K.HasHomology i
    inst✝¹ : ∀ (i : ι), L.HasHomology i
    inst✝ : ∀ (i : ι), M.HasHomology i
    hφ : QuasiIso φ
    ⊢ Iff (QuasiIso (CategoryTheory.CategoryStruct.comp φ φ')) (QuasiIso φ')
  -/
  simp only [quasiIso_iff, quasiIsoAt_iff_comp_left φ φ']
  /-
    🎉 no goals
  -/


lemma quasiIso_of_comp_left (φ : K ⟶ L) (φ' : L ⟶ M) [∀ i, K.HasHomology i]
    [∀ i, L.HasHomology i] [∀ i, M.HasHomology i]
    [hφ : QuasiIso φ] [hφφ' : QuasiIso (φ ≫ φ')] :
    QuasiIso φ' := by
  /-
    ι : Type u_1
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K L M : HomologicalComplex C c
    φ : Quiver.Hom K L
    φ' : Quiver.Hom L M
    inst✝² : ∀ (i : ι), K.HasHomology i
    inst✝¹ : ∀ (i : ι), L.HasHomology i
    inst✝ : ∀ (i : ι), M.HasHomology i
    hφ : QuasiIso φ
    hφφ' : QuasiIso (CategoryTheory.CategoryStruct.comp φ φ')
    ⊢ QuasiIso φ'
  -/
  rw [← quasiIso_iff_comp_left φ φ']
  /-
    ι : Type u_1
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K L M : HomologicalComplex C c
    φ : Quiver.Hom K L
    φ' : Quiver.Hom L M
    inst✝² : ∀ (i : ι), K.HasHomology i
    inst✝¹ : ∀ (i : ι), L.HasHomology i
    inst✝ : ∀ (i : ι), M.HasHomology i
    hφ : QuasiIso φ
    hφφ' : QuasiIso (CategoryTheory.CategoryStruct.comp φ φ')
    ⊢ QuasiIso (CategoryTheory.CategoryStruct.comp φ φ')
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma quasiIsoAt_of_comp_right (φ : K ⟶ L) (φ' : L ⟶ M) (i : ι) [K.HasHomology i]
    [L.HasHomology i] [M.HasHomology i]
    [hφ' : QuasiIsoAt φ' i] [hφφ' : QuasiIsoAt (φ ≫ φ') i] :
    QuasiIsoAt φ i := by
  /-
    ι : Type u_1
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K L M : HomologicalComplex C c
    φ : Quiver.Hom K L
    φ' : Quiver.Hom L M
    i : ι
    inst✝² : K.HasHomology i
    inst✝¹ : L.HasHomology i
    inst✝ : M.HasHomology i
    hφ' : QuasiIsoAt φ' i
    hφφ' : QuasiIsoAt (CategoryTheory.CategoryStruct.comp φ φ') i
    ⊢ QuasiIsoAt φ i
  -/
  rw [quasiIsoAt_iff_isIso_homologyMap] at hφ' hφφ' ⊢
  /-
    ι : Type u_1
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K L M : HomologicalComplex C c
    φ : Quiver.Hom K L
    φ' : Quiver.Hom L M
    i : ι
    inst✝² : K.HasHomology i
    inst✝¹ : L.HasHomology i
    inst✝ : M.HasHomology i
    hφ' : CategoryTheory.IsIso (HomologicalComplex.homologyMap φ' i)
    hφφ' : CategoryTheory.IsIso (HomologicalComplex.homologyMap (CategoryTheory.Ca …
    ⊢ CategoryTheory.IsIso (HomologicalComplex.homologyMap φ i)
  -/
  rw [homologyMap_comp] at hφφ'
  /-
    ι : Type u_1
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K L M : HomologicalComplex C c
    φ : Quiver.Hom K L
    φ' : Quiver.Hom L M
    i : ι
    inst✝² : K.HasHomology i
    inst✝¹ : L.HasHomology i
    inst✝ : M.HasHomology i
    hφ' : CategoryTheory.IsIso (HomologicalComplex.homologyMap φ' i)
    hφφ' : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (HomologicalCo …
    ⊢ CategoryTheory.IsIso (HomologicalComplex.homologyMap φ i)
  -/
  exact IsIso.of_isIso_comp_right (homologyMap φ i) (homologyMap φ' i)
  /-
    🎉 no goals
  -/


lemma quasiIsoAt_iff_comp_right (φ : K ⟶ L) (φ' : L ⟶ M) (i : ι) [K.HasHomology i]
    [L.HasHomology i] [M.HasHomology i]
    [hφ' : QuasiIsoAt φ' i] :
    QuasiIsoAt (φ ≫ φ') i ↔ QuasiIsoAt φ i := by
  /-
    ι : Type u_1
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K L M : HomologicalComplex C c
    φ : Quiver.Hom K L
    φ' : Quiver.Hom L M
    i : ι
    inst✝² : K.HasHomology i
    inst✝¹ : L.HasHomology i
    inst✝ : M.HasHomology i
    hφ' : QuasiIsoAt φ' i
    ⊢ Iff (QuasiIsoAt (CategoryTheory.CategoryStruct.comp φ φ') i) (QuasiIsoAt φ i)
  -/
  constructor
    /-
      case mp
      ι : Type u_1
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      c : ComplexShape ι
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      i : ι
      inst✝² : K.HasHomology i
      inst✝¹ : L.HasHomology i
      inst✝ : M.HasHomology i
      hφ' : QuasiIsoAt φ' i
      ⊢ QuasiIsoAt (CategoryTheory.CategoryStruct.comp φ φ') i → QuasiIsoAt φ i
    -/
  · intro
    /-
      case mp
      ι : Type u_1
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      c : ComplexShape ι
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      i : ι
      inst✝² : K.HasHomology i
      inst✝¹ : L.HasHomology i
      inst✝ : M.HasHomology i
      hφ' : QuasiIsoAt φ' i
      a✝ : QuasiIsoAt (CategoryTheory.CategoryStruct.comp φ φ') i
      ⊢ QuasiIsoAt φ i
    -/
    exact quasiIsoAt_of_comp_right φ φ' i
    /-
      🎉 no goals
    -/
    /-
      case mpr
      ι : Type u_1
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      c : ComplexShape ι
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      i : ι
      inst✝² : K.HasHomology i
      inst✝¹ : L.HasHomology i
      inst✝ : M.HasHomology i
      hφ' : QuasiIsoAt φ' i
      ⊢ QuasiIsoAt φ i → QuasiIsoAt (CategoryTheory.CategoryStruct.comp φ φ') i
    -/
  · intro
    /-
      case mpr
      ι : Type u_1
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      c : ComplexShape ι
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      i : ι
      inst✝² : K.HasHomology i
      inst✝¹ : L.HasHomology i
      inst✝ : M.HasHomology i
      hφ' : QuasiIsoAt φ' i
      a✝ : QuasiIsoAt φ i
      ⊢ QuasiIsoAt (CategoryTheory.CategoryStruct.comp φ φ') i
    -/
    infer_instance
    /-
      🎉 no goals
    -/


lemma quasiIso_iff_comp_right (φ : K ⟶ L) (φ' : L ⟶ M) [∀ i, K.HasHomology i]
    [∀ i, L.HasHomology i] [∀ i, M.HasHomology i]
    [hφ' : QuasiIso φ'] :
    QuasiIso (φ ≫ φ') ↔ QuasiIso φ := by
  /-
    ι : Type u_1
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K L M : HomologicalComplex C c
    φ : Quiver.Hom K L
    φ' : Quiver.Hom L M
    inst✝² : ∀ (i : ι), K.HasHomology i
    inst✝¹ : ∀ (i : ι), L.HasHomology i
    inst✝ : ∀ (i : ι), M.HasHomology i
    hφ' : QuasiIso φ'
    ⊢ Iff (QuasiIso (CategoryTheory.CategoryStruct.comp φ φ')) (QuasiIso φ)
  -/
  simp only [quasiIso_iff, quasiIsoAt_iff_comp_right φ φ']
  /-
    🎉 no goals
  -/


lemma quasiIso_of_comp_right (φ : K ⟶ L) (φ' : L ⟶ M) [∀ i, K.HasHomology i]
    [∀ i, L.HasHomology i] [∀ i, M.HasHomology i]
    [hφ : QuasiIso φ'] [hφφ' : QuasiIso (φ ≫ φ')] :
    QuasiIso φ := by
  /-
    ι : Type u_1
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K L M : HomologicalComplex C c
    φ : Quiver.Hom K L
    φ' : Quiver.Hom L M
    inst✝² : ∀ (i : ι), K.HasHomology i
    inst✝¹ : ∀ (i : ι), L.HasHomology i
    inst✝ : ∀ (i : ι), M.HasHomology i
    hφ : QuasiIso φ'
    hφφ' : QuasiIso (CategoryTheory.CategoryStruct.comp φ φ')
    ⊢ QuasiIso φ
  -/
  rw [← quasiIso_iff_comp_right φ φ']
  /-
    ι : Type u_1
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K L M : HomologicalComplex C c
    φ : Quiver.Hom K L
    φ' : Quiver.Hom L M
    inst✝² : ∀ (i : ι), K.HasHomology i
    inst✝¹ : ∀ (i : ι), L.HasHomology i
    inst✝ : ∀ (i : ι), M.HasHomology i
    hφ : QuasiIso φ'
    hφφ' : QuasiIso (CategoryTheory.CategoryStruct.comp φ φ')
    ⊢ QuasiIso (CategoryTheory.CategoryStruct.comp φ φ')
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma quasiIso_iff_of_arrow_mk_iso (φ : K ⟶ L) (φ' : K' ⟶ L') (e : Arrow.mk φ ≅ Arrow.mk φ')
    [∀ i, K.HasHomology i] [∀ i, L.HasHomology i]
    [∀ i, K'.HasHomology i] [∀ i, L'.HasHomology i] :
    QuasiIso φ ↔ QuasiIso φ' := by
  rw [← quasiIso_iff_comp_left (show K' ⟶ K from e.inv.left) φ,
    ← quasiIso_iff_comp_right φ' (show L' ⟶ L from e.inv.right)]
  /-
    ι : Type u_1
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K L K' L' : HomologicalComplex C c
    φ : Quiver.Hom K L
    φ' : Quiver.Hom K' L'
    e : CategoryTheory.Iso (CategoryTheory.Arrow.mk φ) (CategoryTheory.Arrow.mk φ')
    inst✝³ : ∀ (i : ι), K.HasHomology i
    inst✝² : ∀ (i : ι), L.HasHomology i
    inst✝¹ : ∀ (i : ι), K'.HasHomology i
    inst✝ : ∀ (i : ι), L'.HasHomology i
    ⊢ Iff (QuasiIso (CategoryTheory.CategoryStruct.comp (letFun e.inv.left fun thi …
  -/
  erw [Arrow.w e.inv]
  /-
    ι : Type u_1
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K L K' L' : HomologicalComplex C c
    φ : Quiver.Hom K L
    φ' : Quiver.Hom K' L'
    e : CategoryTheory.Iso (CategoryTheory.Arrow.mk φ) (CategoryTheory.Arrow.mk φ')
    inst✝³ : ∀ (i : ι), K.HasHomology i
    inst✝² : ∀ (i : ι), L.HasHomology i
    inst✝¹ : ∀ (i : ι), K'.HasHomology i
    inst✝ : ∀ (i : ι), L'.HasHomology i
    ⊢ Iff (QuasiIso (CategoryTheory.CategoryStruct.comp (CategoryTheory.Arrow.mk φ …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma quasiIso_of_arrow_mk_iso (φ : K ⟶ L) (φ' : K' ⟶ L') (e : Arrow.mk φ ≅ Arrow.mk φ')
    [∀ i, K.HasHomology i] [∀ i, L.HasHomology i]
    [∀ i, K'.HasHomology i] [∀ i, L'.HasHomology i]
    [hφ : QuasiIso φ] : QuasiIso φ' := by
  /-
    ι : Type u_1
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K L K' L' : HomologicalComplex C c
    φ : Quiver.Hom K L
    φ' : Quiver.Hom K' L'
    e : CategoryTheory.Iso (CategoryTheory.Arrow.mk φ) (CategoryTheory.Arrow.mk φ')
    inst✝³ : ∀ (i : ι), K.HasHomology i
    inst✝² : ∀ (i : ι), L.HasHomology i
    inst✝¹ : ∀ (i : ι), K'.HasHomology i
    inst✝ : ∀ (i : ι), L'.HasHomology i
    hφ : QuasiIso φ
    ⊢ QuasiIso φ'
  -/
  simpa only [← quasiIso_iff_of_arrow_mk_iso φ φ' e]
  /-
    🎉 no goals
  -/


instance quasiIsoAt_map_of_preservesHomology [hφ : QuasiIsoAt φ i] :
    QuasiIsoAt ((F.mapHomologicalComplex c).map φ) i := by
  /-
    ι : Type u_1
    C : Type u
    inst✝¹¹ : CategoryTheory.Category.{v, u} C
    inst✝¹⁰ : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K✝ L✝ M K' L' : HomologicalComplex C c
    C₁ : Type u_2
    C₂ : Type u_3
    inst✝⁹ : CategoryTheory.Category.{u_4, u_2} C₁
    inst✝⁸ : CategoryTheory.Category.{u_5, u_3} C₂
    inst✝⁷ : CategoryTheory.Preadditive C₁
    inst✝⁶ : CategoryTheory.Preadditive C₂
    K L : HomologicalComplex C₁ c
    φ : Quiver.Hom K L
    F : CategoryTheory.Functor C₁ C₂
    inst✝⁵ : F.Additive
    inst✝⁴ : F.PreservesHomology
    i : ι
    inst✝³ : K.HasHomology i
    inst✝² : L.HasHomology i
    inst✝¹ : ((F.mapHomologicalComplex c).obj K).HasHomology i
    inst✝ : ((F.mapHomologicalComplex c).obj L).HasHomology i
    hφ : QuasiIsoAt φ i
    ⊢ QuasiIsoAt ((F.mapHomologicalComplex c).map φ) i
  -/
  rw [quasiIsoAt_iff] at hφ ⊢
  exact ShortComplex.quasiIso_map_of_preservesLeftHomology F
    ((shortComplexFunctor C₁ c i).map φ)


lemma quasiIsoAt_map_iff_of_preservesHomology [F.ReflectsIsomorphisms] :
    QuasiIsoAt ((F.mapHomologicalComplex c).map φ) i ↔ QuasiIsoAt φ i := by
  /-
    ι : Type u_1
    c : ComplexShape ι
    C₁ : Type u_2
    C₂ : Type u_3
    inst✝¹⁰ : CategoryTheory.Category.{u_4, u_2} C₁
    inst✝⁹ : CategoryTheory.Category.{u_5, u_3} C₂
    inst✝⁸ : CategoryTheory.Preadditive C₁
    inst✝⁷ : CategoryTheory.Preadditive C₂
    K L : HomologicalComplex C₁ c
    φ : Quiver.Hom K L
    F : CategoryTheory.Functor C₁ C₂
    inst✝⁶ : F.Additive
    inst✝⁵ : F.PreservesHomology
    i : ι
    inst✝⁴ : K.HasHomology i
    inst✝³ : L.HasHomology i
    inst✝² : ((F.mapHomologicalComplex c).obj K).HasHomology i
    inst✝¹ : ((F.mapHomologicalComplex c).obj L).HasHomology i
    inst✝ : F.ReflectsIsomorphisms
    ⊢ Iff (QuasiIsoAt ((F.mapHomologicalComplex c).map φ) i) (QuasiIsoAt φ i)
  -/
  simp only [quasiIsoAt_iff]
  exact ShortComplex.quasiIso_map_iff_of_preservesLeftHomology F
    ((shortComplexFunctor C₁ c i).map φ)


instance quasiIso_map_of_preservesHomology [hφ : QuasiIso φ] :
    QuasiIso ((F.mapHomologicalComplex c).map φ) where


lemma quasiIso_map_iff_of_preservesHomology [F.ReflectsIsomorphisms] :
    QuasiIso ((F.mapHomologicalComplex c).map φ) ↔ QuasiIso φ := by
  /-
    ι : Type u_1
    c : ComplexShape ι
    C₁ : Type u_2
    C₂ : Type u_3
    inst✝¹⁰ : CategoryTheory.Category.{u_4, u_2} C₁
    inst✝⁹ : CategoryTheory.Category.{u_5, u_3} C₂
    inst✝⁸ : CategoryTheory.Preadditive C₁
    inst✝⁷ : CategoryTheory.Preadditive C₂
    K L : HomologicalComplex C₁ c
    φ : Quiver.Hom K L
    F : CategoryTheory.Functor C₁ C₂
    inst✝⁶ : F.Additive
    inst✝⁵ : F.PreservesHomology
    inst✝⁴ : ∀ (i : ι), K.HasHomology i
    inst✝³ : ∀ (i : ι), L.HasHomology i
    inst✝² : ∀ (i : ι), ((F.mapHomologicalComplex c).obj K).HasHomology i
    inst✝¹ : ∀ (i : ι), ((F.mapHomologicalComplex c).obj L).HasHomology i
    inst✝ : F.ReflectsIsomorphisms
    ⊢ Iff (QuasiIso ((F.mapHomologicalComplex c).map φ)) (QuasiIso φ)
  -/
  simp only [quasiIso_iff, quasiIsoAt_map_iff_of_preservesHomology φ F]
  /-
    🎉 no goals
  -/


/-- The morphism property on `HomologicalComplex C c` given by quasi-isomorphisms. -/
def quasiIso [CategoryWithHomology C] :
    MorphismProperty (HomologicalComplex C c) := fun _ _ f => QuasiIso f


@[simp]
                                                                                                /-
                                                                                                  ι : Type u_1
                                                                                                  C : Type u
                                                                                                  inst✝² : CategoryTheory.Category.{v, u} C
                                                                                                  inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                                                  c : ComplexShape ι
                                                                                                  K L : HomologicalComplex C c
                                                                                                  inst✝ : CategoryTheory.CategoryWithHomology C
                                                                                                  f : Quiver.Hom K L
                                                                                                  ⊢ Iff (HomologicalComplex.quasiIso C c f) (QuasiIso f)
                                                                                                -/
lemma mem_quasiIso_iff [CategoryWithHomology C] (f : K ⟶ L) : quasiIso C c f ↔ QuasiIso f := by rfl
                                                                                                /-
                                                                                                  🎉 no goals
                                                                                                -/


instance : QuasiIso e.hom where
  quasiIsoAt n := by
    classical
    rw [quasiIsoAt_iff_isIso_homologyMap]
    exact (e.toHomologyIso n).isIso_hom


instance : QuasiIso e.inv := (inferInstance : QuasiIso e.symm.hom)


lemma homotopyEquivalences_le_quasiIso [CategoryWithHomology C] :
    homotopyEquivalences C c ≤ quasiIso C c := by
  /-
    ι : Type u_1
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Preadditive C
    c : ComplexShape ι
    inst✝ : CategoryTheory.CategoryWithHomology C
    ⊢ LE.le (HomologicalComplex.homotopyEquivalences C c) (HomologicalComplex.quas …
  -/
  rintro K L _ ⟨e, rfl⟩
  /-
    case intro
    ι : Type u_1
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Preadditive C
    c : ComplexShape ι
    inst✝ : CategoryTheory.CategoryWithHomology C
    K L : HomologicalComplex C c
    e : HomotopyEquiv K L
    ⊢ HomologicalComplex.quasiIso C c e.hom
  -/
  simp only [HomologicalComplex.mem_quasiIso_iff]
  /-
    case intro
    ι : Type u_1
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Preadditive C
    c : ComplexShape ι
    inst✝ : CategoryTheory.CategoryWithHomology C
    K L : HomologicalComplex C c
    e : HomotopyEquiv K L
    ⊢ QuasiIso e.hom
  -/
  infer_instance
  /-
    🎉 no goals
  -/


