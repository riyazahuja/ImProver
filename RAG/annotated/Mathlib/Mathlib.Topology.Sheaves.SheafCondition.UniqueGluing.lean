/-- A family of sections `sf` is compatible, if the restrictions of `sf i` and `sf j` to `U i ⊓ U j`
agree, for all `i` and `j`
-/
def IsCompatible (sf : ∀ i : ι, F.obj (op (U i))) : Prop :=
  ∀ i j : ι, F.map (infLELeft (U i) (U j)).op (sf i) = F.map (infLERight (U i) (U j)).op (sf j)


/-- A section `s` is a gluing for a family of sections `sf` if it restricts to `sf i` on `U i`,
for all `i`
-/
def IsGluing (sf : ∀ i : ι, F.obj (op (U i))) (s : F.obj (op (iSup U))) : Prop :=
  ∀ i : ι, F.map (Opens.leSupr U i).op s = sf i


/--
The sheaf condition in terms of unique gluings. A presheaf `F : Presheaf C X` satisfies this sheaf
condition if and only if, for every compatible family of sections `sf : Π i : ι, F.obj (op (U i))`,
there exists a unique gluing `s : F.obj (op (iSup U))`.

We prove this to be equivalent to the usual one below in
`TopCat.Presheaf.isSheaf_iff_isSheafUniqueGluing`
-/
def IsSheafUniqueGluing : Prop :=
  ∀ ⦃ι : Type x⦄ (U : ι → Opens X) (sf : ∀ i : ι, F.obj (op (U i))),
    IsCompatible F U sf → ∃! s : F.obj (op (iSup U)), IsGluing F U sf s


/-- Given sections over a family of open sets, extend it to include
  sections over pairwise intersections of the open sets. -/
def objPairwiseOfFamily (sf : ∀ i, F.obj (op (U i))) :
    ∀ i, ((Pairwise.diagram U).op ⋙ F).obj i
  | ⟨Pairwise.single i⟩ => sf i
  | ⟨Pairwise.pair i j⟩ => F.map (infLELeft (U i) (U j)).op (sf i)


/-- Given a compatible family of sections over open sets, extend it to a
  section of the functor `(Pairwise.diagram U).op ⋙ F`. -/
def IsCompatible.sectionPairwise {sf} (h : IsCompatible F U sf) :
    ((Pairwise.diagram U).op ⋙ F).sections := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ConcreteCategory C
    X : TopCat
    F : TopCat.Presheaf (Type u) X
    ι : Type x
    U : ι → TopologicalSpace.Opens ↑X
    sf : (i : ι) → (CategoryTheory.forget (Type u)).obj (F.obj { unop := U i })
    h : F.IsCompatible U sf
    ⊢ ↑((CategoryTheory.Pairwise.diagram U).op.comp F).sections
  -/
  refine ⟨objPairwiseOfFamily sf, ?_⟩
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ConcreteCategory C
    X : TopCat
    F : TopCat.Presheaf (Type u) X
    ι : Type x
    U : ι → TopologicalSpace.Opens ↑X
    sf : (i : ι) → (CategoryTheory.forget (Type u)).obj (F.obj { unop := U i })
    h : F.IsCompatible U sf
    ⊢ Membership.mem ((CategoryTheory.Pairwise.diagram U).op.comp F).sections (Top …
  -/
  let G := (Pairwise.diagram U).op ⋙ F
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ConcreteCategory C
    X : TopCat
    F : TopCat.Presheaf (Type u) X
    ι : Type x
    U : ι → TopologicalSpace.Opens ↑X
    sf : (i : ι) → (CategoryTheory.forget (Type u)).obj (F.obj { unop := U i })
    h : F.IsCompatible U sf
    G : CategoryTheory.Functor (Opposite (CategoryTheory.Pairwise ι)) (Type u) :=  …
    ⊢ Membership.mem ((CategoryTheory.Pairwise.diagram U).op.comp F).sections (Top …
  -/
  rintro (i|⟨i,j⟩) (i'|⟨i',j'⟩) (_|_|_|_)
    /-
      case op.single.op.single.op.id_single
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      X : TopCat
      F : TopCat.Presheaf (Type u) X
      ι : Type x
      U : ι → TopologicalSpace.Opens ↑X
      sf : (i : ι) → (CategoryTheory.forget (Type u)).obj (F.obj { unop := U i })
      h : F.IsCompatible U sf
      G : CategoryTheory.Functor (Opposite (CategoryTheory.Pairwise ι)) (Type u) :=  …
      i : ι
      ⊢ Eq (((CategoryTheory.Pairwise.diagram U).op.comp F).map { unop := CategoryTh …
    -/
  · exact congr_fun (G.map_id <| op <| Pairwise.single i) _
    /-
      🎉 no goals
    -/
    /-
      case op.single.op.pair.op.left
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      X : TopCat
      F : TopCat.Presheaf (Type u) X
      ι : Type x
      U : ι → TopologicalSpace.Opens ↑X
      sf : (i : ι) → (CategoryTheory.forget (Type u)).obj (F.obj { unop := U i })
      h : F.IsCompatible U sf
      G : CategoryTheory.Functor (Opposite (CategoryTheory.Pairwise ι)) (Type u) :=  …
      i j' : ι
      ⊢ Eq (((CategoryTheory.Pairwise.diagram U).op.comp F).map { unop := CategoryTh …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case op.single.op.pair.op.right
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      X : TopCat
      F : TopCat.Presheaf (Type u) X
      ι : Type x
      U : ι → TopologicalSpace.Opens ↑X
      sf : (i : ι) → (CategoryTheory.forget (Type u)).obj (F.obj { unop := U i })
      h : F.IsCompatible U sf
      G : CategoryTheory.Functor (Opposite (CategoryTheory.Pairwise ι)) (Type u) :=  …
      i i' : ι
      ⊢ Eq (((CategoryTheory.Pairwise.diagram U).op.comp F).map { unop := CategoryTh …
    -/
  · exact (h i' i).symm
    /-
      🎉 no goals
    -/
    /-
      case op.pair.op.pair.op.id_pair
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      X : TopCat
      F : TopCat.Presheaf (Type u) X
      ι : Type x
      U : ι → TopologicalSpace.Opens ↑X
      sf : (i : ι) → (CategoryTheory.forget (Type u)).obj (F.obj { unop := U i })
      h : F.IsCompatible U sf
      G : CategoryTheory.Functor (Opposite (CategoryTheory.Pairwise ι)) (Type u) :=  …
      i j : ι
      ⊢ Eq (((CategoryTheory.Pairwise.diagram U).op.comp F).map { unop := CategoryTh …
    -/
  · exact congr_fun (G.map_id <| op <| Pairwise.pair i j) _
    /-
      🎉 no goals
    -/


theorem isGluing_iff_pairwise {sf s} : IsGluing F U sf s ↔
    ∀ i, (F.mapCone (Pairwise.cocone U).op).π.app i s = objPairwiseOfFamily sf i := by
  /-
    X : TopCat
    F : TopCat.Presheaf (Type u) X
    ι : Type x
    U : ι → TopologicalSpace.Opens ↑X
    sf : (i : ι) → (CategoryTheory.forget (Type u)).obj (F.obj { unop := U i })
    s : (CategoryTheory.forget (Type u)).obj (F.obj { unop := iSup U })
    ⊢ Iff (F.IsGluing U sf s) (∀ (i : Opposite (CategoryTheory.Pairwise ι)), Eq (( …
  -/
  refine ⟨fun h ↦ ?_, fun h i ↦ h (op <| Pairwise.single i)⟩
  /-
    X : TopCat
    F : TopCat.Presheaf (Type u) X
    ι : Type x
    U : ι → TopologicalSpace.Opens ↑X
    sf : (i : ι) → (CategoryTheory.forget (Type u)).obj (F.obj { unop := U i })
    s : (CategoryTheory.forget (Type u)).obj (F.obj { unop := iSup U })
    h : F.IsGluing U sf s
    ⊢ ∀ (i : Opposite (CategoryTheory.Pairwise ι)), Eq ((CategoryTheory.Functor.ma …
  -/
  rintro (i|⟨i,j⟩)
    /-
      case op.single
      X : TopCat
      F : TopCat.Presheaf (Type u) X
      ι : Type x
      U : ι → TopologicalSpace.Opens ↑X
      sf : (i : ι) → (CategoryTheory.forget (Type u)).obj (F.obj { unop := U i })
      s : (CategoryTheory.forget (Type u)).obj (F.obj { unop := iSup U })
      h : F.IsGluing U sf s
      i : ι
      ⊢ Eq ((CategoryTheory.Functor.mapCone F (CategoryTheory.Pairwise.cocone U).op) …
    -/
  · exact h i
    /-
      🎉 no goals
    -/
    /-
      case op.pair
      X : TopCat
      F : TopCat.Presheaf (Type u) X
      ι : Type x
      U : ι → TopologicalSpace.Opens ↑X
      sf : (i : ι) → (CategoryTheory.forget (Type u)).obj (F.obj { unop := U i })
      s : (CategoryTheory.forget (Type u)).obj (F.obj { unop := iSup U })
      h : F.IsGluing U sf s
      i j : ι
      ⊢ Eq ((CategoryTheory.Functor.mapCone F (CategoryTheory.Pairwise.cocone U).op) …
    -/
  · rw [← (F.mapCone (Pairwise.cocone U).op).w (op <| Pairwise.Hom.left i j)]
    /-
      case op.pair
      X : TopCat
      F : TopCat.Presheaf (Type u) X
      ι : Type x
      U : ι → TopologicalSpace.Opens ↑X
      sf : (i : ι) → (CategoryTheory.forget (Type u)).obj (F.obj { unop := U i })
      s : (CategoryTheory.forget (Type u)).obj (F.obj { unop := iSup U })
      h : F.IsGluing U sf s
      i j : ι
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.mapCone F (C …
    -/
    exact congr_arg _ (h i)
    /-
      🎉 no goals
    -/


/-- For type-valued presheaves, the sheaf condition in terms of unique gluings is equivalent to the
usual sheaf condition.
-/
theorem isSheaf_iff_isSheafUniqueGluing_types : F.IsSheaf ↔ F.IsSheafUniqueGluing := by
  simp_rw [isSheaf_iff_isSheafPairwiseIntersections, IsSheafPairwiseIntersections,
    Types.isLimit_iff, IsSheafUniqueGluing, isGluing_iff_pairwise]
  /-
    X : TopCat
    F : TopCat.Presheaf (Type u) X
    ⊢ Iff (∀ ⦃ι : Type x⦄ (U : ι → TopologicalSpace.Opens ↑X) (s : (j : Opposite ( …
  -/
  refine forall₂_congr fun ι U ↦ ⟨fun h sf cpt ↦ ?_, fun h s hs ↦ ?_⟩
    /-
      case refine_1
      X : TopCat
      F : TopCat.Presheaf (Type u) X
      ι : Type x
      U : ι → TopologicalSpace.Opens ↑X
      h : ∀ (s : (j : Opposite (CategoryTheory.Pairwise ι)) → ((CategoryTheory.Pairw …
      sf : (i : ι) → (CategoryTheory.forget (Type u)).obj (F.obj { unop := U i })
      cpt : F.IsCompatible U sf
      ⊢ ExistsUnique fun s => ∀ (i : Opposite (CategoryTheory.Pairwise ι)), Eq ((Cat …
    -/
  · exact h _ cpt.sectionPairwise.prop
    /-
      🎉 no goals
    -/
  · specialize h (fun i ↦ s <| op <| Pairwise.single i) fun i j ↦
      (hs <| op <| Pairwise.Hom.left i j).trans (hs <| op <| Pairwise.Hom.right i j).symm
    /-
      case refine_2
      X : TopCat
      F : TopCat.Presheaf (Type u) X
      ι : Type x
      U : ι → TopologicalSpace.Opens ↑X
      s : (j : Opposite (CategoryTheory.Pairwise ι)) → ((CategoryTheory.Pairwise.dia …
      hs : Membership.mem ((CategoryTheory.Pairwise.diagram U).op.comp F).sections s
      h : ExistsUnique fun s_1 => ∀ (i : Opposite (CategoryTheory.Pairwise ι)), Eq ( …
      ⊢ ExistsUnique fun x => ∀ (j : Opposite (CategoryTheory.Pairwise ι)), Eq ((Cat …
    -/
    convert h; ext (i|⟨i,j⟩)
      /-
        case h.e'_2.h.h.h.h.e'_3.h.e.h.op.single
        X : TopCat
        F : TopCat.Presheaf (Type u) X
        ι : Type x
        U : ι → TopologicalSpace.Opens ↑X
        s : (j : Opposite (CategoryTheory.Pairwise ι)) → ((CategoryTheory.Pairwise.dia …
        hs : Membership.mem ((CategoryTheory.Pairwise.diagram U).op.comp F).sections s
        h : ExistsUnique fun s_1 => ∀ (i : Opposite (CategoryTheory.Pairwise ι)), Eq ( …
        e_1✝ : Eq (CategoryTheory.Functor.mapCone F (CategoryTheory.Pairwise.cocone U) …
        x✝ : (CategoryTheory.Functor.mapCone F (CategoryTheory.Pairwise.cocone U).op).pt
        a✝ : Opposite (CategoryTheory.Pairwise ι)
        i : ι
        ⊢ Eq (s { unop := CategoryTheory.Pairwise.single i }) (TopCat.Presheaf.objPair …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case h.e'_2.h.h.h.h.e'_3.h.e.h.op.pair
        X : TopCat
        F : TopCat.Presheaf (Type u) X
        ι : Type x
        U : ι → TopologicalSpace.Opens ↑X
        s : (j : Opposite (CategoryTheory.Pairwise ι)) → ((CategoryTheory.Pairwise.dia …
        hs : Membership.mem ((CategoryTheory.Pairwise.diagram U).op.comp F).sections s
        h : ExistsUnique fun s_1 => ∀ (i : Opposite (CategoryTheory.Pairwise ι)), Eq ( …
        e_1✝ : Eq (CategoryTheory.Functor.mapCone F (CategoryTheory.Pairwise.cocone U) …
        x✝ : (CategoryTheory.Functor.mapCone F (CategoryTheory.Pairwise.cocone U).op).pt
        a✝ : Opposite (CategoryTheory.Pairwise ι)
        i j : ι
        ⊢ Eq (s { unop := CategoryTheory.Pairwise.pair i j }) (TopCat.Presheaf.objPair …
      -/
    · exact (hs <| op <| Pairwise.Hom.left i j).symm
      /-
        🎉 no goals
      -/


/-- The usual sheaf condition can be obtained from the sheaf condition
in terms of unique gluings.
-/
theorem isSheaf_of_isSheafUniqueGluing_types (Fsh : F.IsSheafUniqueGluing) : F.IsSheaf :=
  (isSheaf_iff_isSheafUniqueGluing_types F).mpr Fsh


/-- For presheaves valued in a concrete category, whose forgetful functor reflects isomorphisms and
preserves limits, the sheaf condition in terms of unique gluings is equivalent to the usual one.
-/
theorem isSheaf_iff_isSheafUniqueGluing : F.IsSheaf ↔ F.IsSheafUniqueGluing :=
  Iff.trans (isSheaf_iff_isSheaf_comp (forget C) F)
    (isSheaf_iff_isSheafUniqueGluing_types (F ⋙ forget C))


/-- A more convenient way of obtaining a unique gluing of sections for a sheaf.
-/
theorem existsUnique_gluing (sf : ∀ i : ι, F.1.obj (op (U i))) (h : IsCompatible F.1 U sf) :
    ∃! s : F.1.obj (op (iSup U)), IsGluing F.1 U sf s :=
  (isSheaf_iff_isSheafUniqueGluing F.1).mp F.cond U sf h


/-- In this version of the lemma, the inclusion homs `iUV` can be specified directly by the user,
which can be more convenient in practice.
-/
theorem existsUnique_gluing' (V : Opens X) (iUV : ∀ i : ι, U i ⟶ V) (hcover : V ≤ iSup U)
    (sf : ∀ i : ι, F.1.obj (op (U i))) (h : IsCompatible F.1 U sf) :
    ∃! s : F.1.obj (op V), ∀ i : ι, F.1.map (iUV i).op s = sf i := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ConcreteCategory C
    inst✝² : CategoryTheory.Limits.HasLimits C
    inst✝¹ : CategoryTheory.ConcreteCategory.forget.ReflectsIsomorphisms
    inst✝ : CategoryTheory.Limits.PreservesLimits CategoryTheory.ConcreteCategory. …
    X : TopCat
    F : TopCat.Sheaf C X
    ι : Type v
    U : ι → TopologicalSpace.Opens ↑X
    V : TopologicalSpace.Opens ↑X
    iUV : (i : ι) → Quiver.Hom (U i) V
    hcover : LE.le V (iSup U)
    sf : (i : ι) → (CategoryTheory.forget C).obj (F.val.obj { unop := U i })
    h : TopCat.Presheaf.IsCompatible F.val U sf
    ⊢ ExistsUnique fun s => ∀ (i : ι), Eq ((F.val.map (iUV i).op) s) (sf i)
  -/
  have V_eq_supr_U : V = iSup U := le_antisymm hcover (iSup_le fun i => (iUV i).le)
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ConcreteCategory C
    inst✝² : CategoryTheory.Limits.HasLimits C
    inst✝¹ : CategoryTheory.ConcreteCategory.forget.ReflectsIsomorphisms
    inst✝ : CategoryTheory.Limits.PreservesLimits CategoryTheory.ConcreteCategory. …
    X : TopCat
    F : TopCat.Sheaf C X
    ι : Type v
    U : ι → TopologicalSpace.Opens ↑X
    V : TopologicalSpace.Opens ↑X
    iUV : (i : ι) → Quiver.Hom (U i) V
    hcover : LE.le V (iSup U)
    sf : (i : ι) → (CategoryTheory.forget C).obj (F.val.obj { unop := U i })
    h : TopCat.Presheaf.IsCompatible F.val U sf
    V_eq_supr_U : Eq V (iSup U)
    ⊢ ExistsUnique fun s => ∀ (i : ι), Eq ((F.val.map (iUV i).op) s) (sf i)
  -/
  obtain ⟨gl, gl_spec, gl_uniq⟩ := F.existsUnique_gluing U sf h
  /-
    case intro.intro
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ConcreteCategory C
    inst✝² : CategoryTheory.Limits.HasLimits C
    inst✝¹ : CategoryTheory.ConcreteCategory.forget.ReflectsIsomorphisms
    inst✝ : CategoryTheory.Limits.PreservesLimits CategoryTheory.ConcreteCategory. …
    X : TopCat
    F : TopCat.Sheaf C X
    ι : Type v
    U : ι → TopologicalSpace.Opens ↑X
    V : TopologicalSpace.Opens ↑X
    iUV : (i : ι) → Quiver.Hom (U i) V
    hcover : LE.le V (iSup U)
    sf : (i : ι) → (CategoryTheory.forget C).obj (F.val.obj { unop := U i })
    h : TopCat.Presheaf.IsCompatible F.val U sf
    V_eq_supr_U : Eq V (iSup U)
    gl : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })
    gl_spec : TopCat.Presheaf.IsGluing F.val U sf gl
    gl_uniq : ∀ (y : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })) …
    ⊢ ExistsUnique fun s => ∀ (i : ι), Eq ((F.val.map (iUV i).op) s) (sf i)
  -/
  refine ⟨F.1.map (eqToHom V_eq_supr_U).op gl, ?_, ?_⟩
    /-
      case intro.intro.refine_1
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.ConcreteCategory C
      inst✝² : CategoryTheory.Limits.HasLimits C
      inst✝¹ : CategoryTheory.ConcreteCategory.forget.ReflectsIsomorphisms
      inst✝ : CategoryTheory.Limits.PreservesLimits CategoryTheory.ConcreteCategory. …
      X : TopCat
      F : TopCat.Sheaf C X
      ι : Type v
      U : ι → TopologicalSpace.Opens ↑X
      V : TopologicalSpace.Opens ↑X
      iUV : (i : ι) → Quiver.Hom (U i) V
      hcover : LE.le V (iSup U)
      sf : (i : ι) → (CategoryTheory.forget C).obj (F.val.obj { unop := U i })
      h : TopCat.Presheaf.IsCompatible F.val U sf
      V_eq_supr_U : Eq V (iSup U)
      gl : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })
      gl_spec : TopCat.Presheaf.IsGluing F.val U sf gl
      gl_uniq : ∀ (y : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })) …
      ⊢ (fun s => ∀ (i : ι), Eq ((F.val.map (iUV i).op) s) (sf i)) ((F.val.map (Cate …
    -/
  · intro i
    /-
      case intro.intro.refine_1
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.ConcreteCategory C
      inst✝² : CategoryTheory.Limits.HasLimits C
      inst✝¹ : CategoryTheory.ConcreteCategory.forget.ReflectsIsomorphisms
      inst✝ : CategoryTheory.Limits.PreservesLimits CategoryTheory.ConcreteCategory. …
      X : TopCat
      F : TopCat.Sheaf C X
      ι : Type v
      U : ι → TopologicalSpace.Opens ↑X
      V : TopologicalSpace.Opens ↑X
      iUV : (i : ι) → Quiver.Hom (U i) V
      hcover : LE.le V (iSup U)
      sf : (i : ι) → (CategoryTheory.forget C).obj (F.val.obj { unop := U i })
      h : TopCat.Presheaf.IsCompatible F.val U sf
      V_eq_supr_U : Eq V (iSup U)
      gl : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })
      gl_spec : TopCat.Presheaf.IsGluing F.val U sf gl
      gl_uniq : ∀ (y : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })) …
      i : ι
      ⊢ Eq ((F.val.map (iUV i).op) ((F.val.map (CategoryTheory.eqToHom V_eq_supr_U). …
    -/
    rw [← comp_apply, ← F.1.map_comp]
    /-
      case intro.intro.refine_1
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.ConcreteCategory C
      inst✝² : CategoryTheory.Limits.HasLimits C
      inst✝¹ : CategoryTheory.ConcreteCategory.forget.ReflectsIsomorphisms
      inst✝ : CategoryTheory.Limits.PreservesLimits CategoryTheory.ConcreteCategory. …
      X : TopCat
      F : TopCat.Sheaf C X
      ι : Type v
      U : ι → TopologicalSpace.Opens ↑X
      V : TopologicalSpace.Opens ↑X
      iUV : (i : ι) → Quiver.Hom (U i) V
      hcover : LE.le V (iSup U)
      sf : (i : ι) → (CategoryTheory.forget C).obj (F.val.obj { unop := U i })
      h : TopCat.Presheaf.IsCompatible F.val U sf
      V_eq_supr_U : Eq V (iSup U)
      gl : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })
      gl_spec : TopCat.Presheaf.IsGluing F.val U sf gl
      gl_uniq : ∀ (y : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })) …
      i : ι
      ⊢ Eq ((F.val.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom V …
    -/
    exact gl_spec i
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.ConcreteCategory C
      inst✝² : CategoryTheory.Limits.HasLimits C
      inst✝¹ : CategoryTheory.ConcreteCategory.forget.ReflectsIsomorphisms
      inst✝ : CategoryTheory.Limits.PreservesLimits CategoryTheory.ConcreteCategory. …
      X : TopCat
      F : TopCat.Sheaf C X
      ι : Type v
      U : ι → TopologicalSpace.Opens ↑X
      V : TopologicalSpace.Opens ↑X
      iUV : (i : ι) → Quiver.Hom (U i) V
      hcover : LE.le V (iSup U)
      sf : (i : ι) → (CategoryTheory.forget C).obj (F.val.obj { unop := U i })
      h : TopCat.Presheaf.IsCompatible F.val U sf
      V_eq_supr_U : Eq V (iSup U)
      gl : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })
      gl_spec : TopCat.Presheaf.IsGluing F.val U sf gl
      gl_uniq : ∀ (y : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })) …
      ⊢ ∀ (y : (CategoryTheory.forget C).obj (F.val.obj { unop := V })), (fun s => ∀ …
    -/
  · intro gl' gl'_spec
    /-
      case intro.intro.refine_2
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.ConcreteCategory C
      inst✝² : CategoryTheory.Limits.HasLimits C
      inst✝¹ : CategoryTheory.ConcreteCategory.forget.ReflectsIsomorphisms
      inst✝ : CategoryTheory.Limits.PreservesLimits CategoryTheory.ConcreteCategory. …
      X : TopCat
      F : TopCat.Sheaf C X
      ι : Type v
      U : ι → TopologicalSpace.Opens ↑X
      V : TopologicalSpace.Opens ↑X
      iUV : (i : ι) → Quiver.Hom (U i) V
      hcover : LE.le V (iSup U)
      sf : (i : ι) → (CategoryTheory.forget C).obj (F.val.obj { unop := U i })
      h : TopCat.Presheaf.IsCompatible F.val U sf
      V_eq_supr_U : Eq V (iSup U)
      gl : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })
      gl_spec : TopCat.Presheaf.IsGluing F.val U sf gl
      gl_uniq : ∀ (y : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })) …
      gl' : (CategoryTheory.forget C).obj (F.val.obj { unop := V })
      gl'_spec : ∀ (i : ι), Eq ((F.val.map (iUV i).op) gl') (sf i)
      ⊢ Eq gl' ((F.val.map (CategoryTheory.eqToHom V_eq_supr_U).op) gl)
    -/
    convert congr_arg _ (gl_uniq (F.1.map (eqToHom V_eq_supr_U.symm).op gl') fun i => _) <;>
      /-
        case h.e'_2
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        inst✝³ : CategoryTheory.ConcreteCategory C
        inst✝² : CategoryTheory.Limits.HasLimits C
        inst✝¹ : CategoryTheory.ConcreteCategory.forget.ReflectsIsomorphisms
        inst✝ : CategoryTheory.Limits.PreservesLimits CategoryTheory.ConcreteCategory. …
        X : TopCat
        F : TopCat.Sheaf C X
        ι : Type v
        U : ι → TopologicalSpace.Opens ↑X
        V : TopologicalSpace.Opens ↑X
        iUV : (i : ι) → Quiver.Hom (U i) V
        hcover : LE.le V (iSup U)
        sf : (i : ι) → (CategoryTheory.forget C).obj (F.val.obj { unop := U i })
        h : TopCat.Presheaf.IsCompatible F.val U sf
        V_eq_supr_U : Eq V (iSup U)
        gl : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })
        gl_spec : TopCat.Presheaf.IsGluing F.val U sf gl
        gl_uniq : ∀ (y : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })) …
        gl' : (CategoryTheory.forget C).obj (F.val.obj { unop := V })
        gl'_spec : ∀ (i : ι), Eq ((F.val.map (iUV i).op) gl') (sf i)
        ⊢ Eq gl' ((F.val.map (CategoryTheory.eqToHom V_eq_supr_U).op) ((F.val.map (Cat …
      -/
      rw [← comp_apply, ← F.1.map_comp]
      /-
        case h.e'_2
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        inst✝³ : CategoryTheory.ConcreteCategory C
        inst✝² : CategoryTheory.Limits.HasLimits C
        inst✝¹ : CategoryTheory.ConcreteCategory.forget.ReflectsIsomorphisms
        inst✝ : CategoryTheory.Limits.PreservesLimits CategoryTheory.ConcreteCategory. …
        X : TopCat
        F : TopCat.Sheaf C X
        ι : Type v
        U : ι → TopologicalSpace.Opens ↑X
        V : TopologicalSpace.Opens ↑X
        iUV : (i : ι) → Quiver.Hom (U i) V
        hcover : LE.le V (iSup U)
        sf : (i : ι) → (CategoryTheory.forget C).obj (F.val.obj { unop := U i })
        h : TopCat.Presheaf.IsCompatible F.val U sf
        V_eq_supr_U : Eq V (iSup U)
        gl : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })
        gl_spec : TopCat.Presheaf.IsGluing F.val U sf gl
        gl_uniq : ∀ (y : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })) …
        gl' : (CategoryTheory.forget C).obj (F.val.obj { unop := V })
        gl'_spec : ∀ (i : ι), Eq ((F.val.map (iUV i).op) gl') (sf i)
        ⊢ Eq gl' ((F.val.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToH …
      -/
    · rw [eqToHom_op, eqToHom_op, eqToHom_trans, eqToHom_refl, F.1.map_id, id_apply]
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.refine_2.convert_3
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        inst✝³ : CategoryTheory.ConcreteCategory C
        inst✝² : CategoryTheory.Limits.HasLimits C
        inst✝¹ : CategoryTheory.ConcreteCategory.forget.ReflectsIsomorphisms
        inst✝ : CategoryTheory.Limits.PreservesLimits CategoryTheory.ConcreteCategory. …
        X : TopCat
        F : TopCat.Sheaf C X
        ι : Type v
        U : ι → TopologicalSpace.Opens ↑X
        V : TopologicalSpace.Opens ↑X
        iUV : (i : ι) → Quiver.Hom (U i) V
        hcover : LE.le V (iSup U)
        sf : (i : ι) → (CategoryTheory.forget C).obj (F.val.obj { unop := U i })
        h : TopCat.Presheaf.IsCompatible F.val U sf
        V_eq_supr_U : Eq V (iSup U)
        gl : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })
        gl_spec : TopCat.Presheaf.IsGluing F.val U sf gl
        gl_uniq : ∀ (y : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })) …
        gl' : (CategoryTheory.forget C).obj (F.val.obj { unop := V })
        gl'_spec : ∀ (i : ι), Eq ((F.val.map (iUV i).op) gl') (sf i)
        i : ι
        ⊢ Eq ((F.val.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯ …
      -/
    · convert gl'_spec i
      /-
        🎉 no goals
      -/


@[ext]
theorem eq_of_locally_eq (s t : F.1.obj (op (iSup U)))
    (h : ∀ i, F.1.map (Opens.leSupr U i).op s = F.1.map (Opens.leSupr U i).op t) : s = t := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ConcreteCategory C
    inst✝² : CategoryTheory.Limits.HasLimits C
    inst✝¹ : CategoryTheory.ConcreteCategory.forget.ReflectsIsomorphisms
    inst✝ : CategoryTheory.Limits.PreservesLimits CategoryTheory.ConcreteCategory. …
    X : TopCat
    F : TopCat.Sheaf C X
    ι : Type v
    U : ι → TopologicalSpace.Opens ↑X
    s t : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })
    h : ∀ (i : ι), Eq ((F.val.map (TopologicalSpace.Opens.leSupr U i).op) s) ((F.v …
    ⊢ Eq s t
  -/
  let sf : ∀ i : ι, F.1.obj (op (U i)) := fun i => F.1.map (Opens.leSupr U i).op s
  have sf_compatible : IsCompatible _ U sf := by
    intro i j
    simp_rw [sf, ← comp_apply, ← F.1.map_comp]
    rfl
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ConcreteCategory C
    inst✝² : CategoryTheory.Limits.HasLimits C
    inst✝¹ : CategoryTheory.ConcreteCategory.forget.ReflectsIsomorphisms
    inst✝ : CategoryTheory.Limits.PreservesLimits CategoryTheory.ConcreteCategory. …
    X : TopCat
    F : TopCat.Sheaf C X
    ι : Type v
    U : ι → TopologicalSpace.Opens ↑X
    s t : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })
    h : ∀ (i : ι), Eq ((F.val.map (TopologicalSpace.Opens.leSupr U i).op) s) ((F.v …
    sf : (i : ι) → (CategoryTheory.forget C).obj (F.val.obj { unop := U i }) := fu …
    sf_compatible : TopCat.Presheaf.IsCompatible F.val U sf
    ⊢ Eq s t
  -/
  obtain ⟨gl, -, gl_uniq⟩ := F.existsUnique_gluing U sf sf_compatible
  /-
    case intro.intro
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ConcreteCategory C
    inst✝² : CategoryTheory.Limits.HasLimits C
    inst✝¹ : CategoryTheory.ConcreteCategory.forget.ReflectsIsomorphisms
    inst✝ : CategoryTheory.Limits.PreservesLimits CategoryTheory.ConcreteCategory. …
    X : TopCat
    F : TopCat.Sheaf C X
    ι : Type v
    U : ι → TopologicalSpace.Opens ↑X
    s t : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })
    h : ∀ (i : ι), Eq ((F.val.map (TopologicalSpace.Opens.leSupr U i).op) s) ((F.v …
    sf : (i : ι) → (CategoryTheory.forget C).obj (F.val.obj { unop := U i }) := fu …
    sf_compatible : TopCat.Presheaf.IsCompatible F.val U sf
    gl : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })
    gl_uniq : ∀ (y : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })) …
    ⊢ Eq s t
  -/
  trans gl
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.ConcreteCategory C
      inst✝² : CategoryTheory.Limits.HasLimits C
      inst✝¹ : CategoryTheory.ConcreteCategory.forget.ReflectsIsomorphisms
      inst✝ : CategoryTheory.Limits.PreservesLimits CategoryTheory.ConcreteCategory. …
      X : TopCat
      F : TopCat.Sheaf C X
      ι : Type v
      U : ι → TopologicalSpace.Opens ↑X
      s t : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })
      h : ∀ (i : ι), Eq ((F.val.map (TopologicalSpace.Opens.leSupr U i).op) s) ((F.v …
      sf : (i : ι) → (CategoryTheory.forget C).obj (F.val.obj { unop := U i }) := fu …
      sf_compatible : TopCat.Presheaf.IsCompatible F.val U sf
      gl : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })
      gl_uniq : ∀ (y : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })) …
      ⊢ Eq s gl
    -/
  · apply gl_uniq
    /-
      case a
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.ConcreteCategory C
      inst✝² : CategoryTheory.Limits.HasLimits C
      inst✝¹ : CategoryTheory.ConcreteCategory.forget.ReflectsIsomorphisms
      inst✝ : CategoryTheory.Limits.PreservesLimits CategoryTheory.ConcreteCategory. …
      X : TopCat
      F : TopCat.Sheaf C X
      ι : Type v
      U : ι → TopologicalSpace.Opens ↑X
      s t : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })
      h : ∀ (i : ι), Eq ((F.val.map (TopologicalSpace.Opens.leSupr U i).op) s) ((F.v …
      sf : (i : ι) → (CategoryTheory.forget C).obj (F.val.obj { unop := U i }) := fu …
      sf_compatible : TopCat.Presheaf.IsCompatible F.val U sf
      gl : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })
      gl_uniq : ∀ (y : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })) …
      ⊢ TopCat.Presheaf.IsGluing F.val U sf s
    -/
    intro i
    /-
      case a
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.ConcreteCategory C
      inst✝² : CategoryTheory.Limits.HasLimits C
      inst✝¹ : CategoryTheory.ConcreteCategory.forget.ReflectsIsomorphisms
      inst✝ : CategoryTheory.Limits.PreservesLimits CategoryTheory.ConcreteCategory. …
      X : TopCat
      F : TopCat.Sheaf C X
      ι : Type v
      U : ι → TopologicalSpace.Opens ↑X
      s t : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })
      h : ∀ (i : ι), Eq ((F.val.map (TopologicalSpace.Opens.leSupr U i).op) s) ((F.v …
      sf : (i : ι) → (CategoryTheory.forget C).obj (F.val.obj { unop := U i }) := fu …
      sf_compatible : TopCat.Presheaf.IsCompatible F.val U sf
      gl : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })
      gl_uniq : ∀ (y : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })) …
      i : ι
      ⊢ Eq ((F.val.map (TopologicalSpace.Opens.leSupr U i).op) s) (sf i)
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.ConcreteCategory C
      inst✝² : CategoryTheory.Limits.HasLimits C
      inst✝¹ : CategoryTheory.ConcreteCategory.forget.ReflectsIsomorphisms
      inst✝ : CategoryTheory.Limits.PreservesLimits CategoryTheory.ConcreteCategory. …
      X : TopCat
      F : TopCat.Sheaf C X
      ι : Type v
      U : ι → TopologicalSpace.Opens ↑X
      s t : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })
      h : ∀ (i : ι), Eq ((F.val.map (TopologicalSpace.Opens.leSupr U i).op) s) ((F.v …
      sf : (i : ι) → (CategoryTheory.forget C).obj (F.val.obj { unop := U i }) := fu …
      sf_compatible : TopCat.Presheaf.IsCompatible F.val U sf
      gl : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })
      gl_uniq : ∀ (y : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })) …
      ⊢ Eq gl t
    -/
  · symm
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.ConcreteCategory C
      inst✝² : CategoryTheory.Limits.HasLimits C
      inst✝¹ : CategoryTheory.ConcreteCategory.forget.ReflectsIsomorphisms
      inst✝ : CategoryTheory.Limits.PreservesLimits CategoryTheory.ConcreteCategory. …
      X : TopCat
      F : TopCat.Sheaf C X
      ι : Type v
      U : ι → TopologicalSpace.Opens ↑X
      s t : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })
      h : ∀ (i : ι), Eq ((F.val.map (TopologicalSpace.Opens.leSupr U i).op) s) ((F.v …
      sf : (i : ι) → (CategoryTheory.forget C).obj (F.val.obj { unop := U i }) := fu …
      sf_compatible : TopCat.Presheaf.IsCompatible F.val U sf
      gl : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })
      gl_uniq : ∀ (y : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })) …
      ⊢ Eq t gl
    -/
    apply gl_uniq
    /-
      case a
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.ConcreteCategory C
      inst✝² : CategoryTheory.Limits.HasLimits C
      inst✝¹ : CategoryTheory.ConcreteCategory.forget.ReflectsIsomorphisms
      inst✝ : CategoryTheory.Limits.PreservesLimits CategoryTheory.ConcreteCategory. …
      X : TopCat
      F : TopCat.Sheaf C X
      ι : Type v
      U : ι → TopologicalSpace.Opens ↑X
      s t : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })
      h : ∀ (i : ι), Eq ((F.val.map (TopologicalSpace.Opens.leSupr U i).op) s) ((F.v …
      sf : (i : ι) → (CategoryTheory.forget C).obj (F.val.obj { unop := U i }) := fu …
      sf_compatible : TopCat.Presheaf.IsCompatible F.val U sf
      gl : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })
      gl_uniq : ∀ (y : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })) …
      ⊢ TopCat.Presheaf.IsGluing F.val U sf t
    -/
    intro i
    /-
      case a
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.ConcreteCategory C
      inst✝² : CategoryTheory.Limits.HasLimits C
      inst✝¹ : CategoryTheory.ConcreteCategory.forget.ReflectsIsomorphisms
      inst✝ : CategoryTheory.Limits.PreservesLimits CategoryTheory.ConcreteCategory. …
      X : TopCat
      F : TopCat.Sheaf C X
      ι : Type v
      U : ι → TopologicalSpace.Opens ↑X
      s t : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })
      h : ∀ (i : ι), Eq ((F.val.map (TopologicalSpace.Opens.leSupr U i).op) s) ((F.v …
      sf : (i : ι) → (CategoryTheory.forget C).obj (F.val.obj { unop := U i }) := fu …
      sf_compatible : TopCat.Presheaf.IsCompatible F.val U sf
      gl : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })
      gl_uniq : ∀ (y : (CategoryTheory.forget C).obj (F.val.obj { unop := iSup U })) …
      i : ι
      ⊢ Eq ((F.val.map (TopologicalSpace.Opens.leSupr U i).op) t) (sf i)
    -/
    rw [← h]
    /-
      🎉 no goals
    -/


/-- In this version of the lemma, the inclusion homs `iUV` can be specified directly by the user,
which can be more convenient in practice.
-/
theorem eq_of_locally_eq' (V : Opens X) (iUV : ∀ i : ι, U i ⟶ V) (hcover : V ≤ iSup U)
    (s t : F.1.obj (op V)) (h : ∀ i, F.1.map (iUV i).op s = F.1.map (iUV i).op t) : s = t := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ConcreteCategory C
    inst✝² : CategoryTheory.Limits.HasLimits C
    inst✝¹ : CategoryTheory.ConcreteCategory.forget.ReflectsIsomorphisms
    inst✝ : CategoryTheory.Limits.PreservesLimits CategoryTheory.ConcreteCategory. …
    X : TopCat
    F : TopCat.Sheaf C X
    ι : Type v
    U : ι → TopologicalSpace.Opens ↑X
    V : TopologicalSpace.Opens ↑X
    iUV : (i : ι) → Quiver.Hom (U i) V
    hcover : LE.le V (iSup U)
    s t : (CategoryTheory.forget C).obj (F.val.obj { unop := V })
    h : ∀ (i : ι), Eq ((F.val.map (iUV i).op) s) ((F.val.map (iUV i).op) t)
    ⊢ Eq s t
  -/
  have V_eq_supr_U : V = iSup U := le_antisymm hcover (iSup_le fun i => (iUV i).le)
  suffices F.1.map (eqToHom V_eq_supr_U.symm).op s = F.1.map (eqToHom V_eq_supr_U.symm).op t by
    convert congr_arg (F.1.map (eqToHom V_eq_supr_U).op) this <;>
    rw [← comp_apply, ← F.1.map_comp, eqToHom_op, eqToHom_op, eqToHom_trans, eqToHom_refl,
      F.1.map_id, id_apply]
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ConcreteCategory C
    inst✝² : CategoryTheory.Limits.HasLimits C
    inst✝¹ : CategoryTheory.ConcreteCategory.forget.ReflectsIsomorphisms
    inst✝ : CategoryTheory.Limits.PreservesLimits CategoryTheory.ConcreteCategory. …
    X : TopCat
    F : TopCat.Sheaf C X
    ι : Type v
    U : ι → TopologicalSpace.Opens ↑X
    V : TopologicalSpace.Opens ↑X
    iUV : (i : ι) → Quiver.Hom (U i) V
    hcover : LE.le V (iSup U)
    s t : (CategoryTheory.forget C).obj (F.val.obj { unop := V })
    h : ∀ (i : ι), Eq ((F.val.map (iUV i).op) s) ((F.val.map (iUV i).op) t)
    V_eq_supr_U : Eq V (iSup U)
    ⊢ Eq ((F.val.map (CategoryTheory.eqToHom ⋯).op) s) ((F.val.map (CategoryTheory …
  -/
  apply eq_of_locally_eq
  /-
    case h
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ConcreteCategory C
    inst✝² : CategoryTheory.Limits.HasLimits C
    inst✝¹ : CategoryTheory.ConcreteCategory.forget.ReflectsIsomorphisms
    inst✝ : CategoryTheory.Limits.PreservesLimits CategoryTheory.ConcreteCategory. …
    X : TopCat
    F : TopCat.Sheaf C X
    ι : Type v
    U : ι → TopologicalSpace.Opens ↑X
    V : TopologicalSpace.Opens ↑X
    iUV : (i : ι) → Quiver.Hom (U i) V
    hcover : LE.le V (iSup U)
    s t : (CategoryTheory.forget C).obj (F.val.obj { unop := V })
    h : ∀ (i : ι), Eq ((F.val.map (iUV i).op) s) ((F.val.map (iUV i).op) t)
    V_eq_supr_U : Eq V (iSup U)
    ⊢ ∀ (i : ι), Eq ((F.val.map (TopologicalSpace.Opens.leSupr U i).op) ((F.val.ma …
  -/
  intro i
  /-
    case h
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ConcreteCategory C
    inst✝² : CategoryTheory.Limits.HasLimits C
    inst✝¹ : CategoryTheory.ConcreteCategory.forget.ReflectsIsomorphisms
    inst✝ : CategoryTheory.Limits.PreservesLimits CategoryTheory.ConcreteCategory. …
    X : TopCat
    F : TopCat.Sheaf C X
    ι : Type v
    U : ι → TopologicalSpace.Opens ↑X
    V : TopologicalSpace.Opens ↑X
    iUV : (i : ι) → Quiver.Hom (U i) V
    hcover : LE.le V (iSup U)
    s t : (CategoryTheory.forget C).obj (F.val.obj { unop := V })
    h : ∀ (i : ι), Eq ((F.val.map (iUV i).op) s) ((F.val.map (iUV i).op) t)
    V_eq_supr_U : Eq V (iSup U)
    i : ι
    ⊢ Eq ((F.val.map (TopologicalSpace.Opens.leSupr U i).op) ((F.val.map (Category …
  -/
  rw [← comp_apply, ← comp_apply, ← F.1.map_comp]
  /-
    case h
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ConcreteCategory C
    inst✝² : CategoryTheory.Limits.HasLimits C
    inst✝¹ : CategoryTheory.ConcreteCategory.forget.ReflectsIsomorphisms
    inst✝ : CategoryTheory.Limits.PreservesLimits CategoryTheory.ConcreteCategory. …
    X : TopCat
    F : TopCat.Sheaf C X
    ι : Type v
    U : ι → TopologicalSpace.Opens ↑X
    V : TopologicalSpace.Opens ↑X
    iUV : (i : ι) → Quiver.Hom (U i) V
    hcover : LE.le V (iSup U)
    s t : (CategoryTheory.forget C).obj (F.val.obj { unop := V })
    h : ∀ (i : ι), Eq ((F.val.map (iUV i).op) s) ((F.val.map (iUV i).op) t)
    V_eq_supr_U : Eq V (iSup U)
    i : ι
    ⊢ Eq ((F.val.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯ …
  -/
  convert h i
  /-
    🎉 no goals
  -/


theorem eq_of_locally_eq₂ {U₁ U₂ V : Opens X} (i₁ : U₁ ⟶ V) (i₂ : U₂ ⟶ V) (hcover : V ≤ U₁ ⊔ U₂)
    (s t : F.1.obj (op V)) (h₁ : F.1.map i₁.op s = F.1.map i₁.op t)
    (h₂ : F.1.map i₂.op s = F.1.map i₂.op t) : s = t := by
  classical
    fapply F.eq_of_locally_eq' fun t : ULift Bool => if t.1 then U₁ else U₂
    · exact fun i => if h : i.1 then eqToHom (if_pos h) ≫ i₁ else eqToHom (if_neg h) ≫ i₂
    · refine le_trans hcover ?_
      rw [sup_le_iff]
      constructor
      · convert le_iSup (fun t : ULift Bool => if t.1 then U₁ else U₂) (ULift.up true)
      · convert le_iSup (fun t : ULift Bool => if t.1 then U₁ else U₂) (ULift.up false)
    · rintro ⟨_ | _⟩
      any_goals exact h₁
      any_goals exact h₂


