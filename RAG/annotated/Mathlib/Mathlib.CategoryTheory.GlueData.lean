/-- A gluing datum consists of
1. An index type `J`
2. An object `U i` for each `i : J`.
3. An object `V i j` for each `i j : J`.
4. A monomorphism `f i j : V i j ⟶ U i` for each `i j : J`.
5. A transition map `t i j : V i j ⟶ V j i` for each `i j : J`.
such that
6. `f i i` is an isomorphism.
7. `t i i` is the identity.
8. The pullback for `f i j` and `f i k` exists.
9. `V i j ×[U i] V i k ⟶ V i j ⟶ V j i` factors through `V j k ×[U j] V j i ⟶ V j i` via some
    `t' : V i j ×[U i] V i k ⟶ V j k ×[U j] V j i`.
10. `t' i j k ≫ t' j k i ≫ t' k i j = 𝟙 _`.
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): linter not ported yet
-- @[nolint has_nonempty_instance]
structure GlueData where
  J : Type v
  U : J → C
  V : J × J → C
  f : ∀ i j, V (i, j) ⟶ U i
  f_mono : ∀ i j, Mono (f i j) := by infer_instance
  f_hasPullback : ∀ i j k, HasPullback (f i j) (f i k) := by infer_instance
  f_id : ∀ i, IsIso (f i i) := by infer_instance
  t : ∀ i j, V (i, j) ⟶ V (j, i)
  t_id : ∀ i, t i i = 𝟙 _
  t' : ∀ i j k, pullback (f i j) (f i k) ⟶ pullback (f j k) (f j i)
  t_fac : ∀ i j k, t' i j k ≫ pullback.snd _ _ = pullback.fst _ _ ≫ t i j
  cocycle : ∀ i j k, t' i j k ≫ t' j k i ≫ t' k i j = 𝟙 _


attribute [reassoc] GlueData.t_fac GlueData.cocycle


@[simp]
theorem t'_iij (i j : D.J) : D.t' i i j = (pullbackSymmetry _ _).hom := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    D : CategoryTheory.GlueData C
    i j : D.J
    ⊢ Eq (D.t' i i j) (CategoryTheory.Limits.pullbackSymmetry (D.f i i) (D.f i j)) …
  -/
  have eq₁ := D.t_fac i i j
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    D : CategoryTheory.GlueData C
    i j : D.J
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp (D.t' i i j) (CategoryTheory.Limi …
    ⊢ Eq (D.t' i i j) (CategoryTheory.Limits.pullbackSymmetry (D.f i i) (D.f i j)) …
  -/
  have eq₂ := (IsIso.eq_comp_inv (D.f i i)).mpr (@pullback.condition _ _ _ _ _ _ (D.f i j) _)
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    D : CategoryTheory.GlueData C
    i j : D.J
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp (D.t' i i j) (CategoryTheory.Limi …
    eq₂ : Eq (CategoryTheory.Limits.pullback.fst (D.f i i) (D.f i j)) (CategoryThe …
    ⊢ Eq (D.t' i i j) (CategoryTheory.Limits.pullbackSymmetry (D.f i i) (D.f i j)) …
  -/
  rw [D.t_id, Category.comp_id, eq₂] at eq₁
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    D : CategoryTheory.GlueData C
    i j : D.J
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp (D.t' i i j) (CategoryTheory.Limi …
    eq₂ : Eq (CategoryTheory.Limits.pullback.fst (D.f i i) (D.f i j)) (CategoryThe …
    ⊢ Eq (D.t' i i j) (CategoryTheory.Limits.pullbackSymmetry (D.f i i) (D.f i j)) …
  -/
  have eq₃ := (IsIso.eq_comp_inv (D.f i i)).mp eq₁
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    D : CategoryTheory.GlueData C
    i j : D.J
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp (D.t' i i j) (CategoryTheory.Limi …
    eq₂ : Eq (CategoryTheory.Limits.pullback.fst (D.f i i) (D.f i j)) (CategoryThe …
    eq₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.co …
    ⊢ Eq (D.t' i i j) (CategoryTheory.Limits.pullbackSymmetry (D.f i i) (D.f i j)) …
  -/
  rw [Category.assoc, ← pullback.condition, ← Category.assoc] at eq₃
  exact
    Mono.right_cancellation _ _
      ((Mono.right_cancellation _ _ eq₃).trans (pullbackSymmetry_hom_comp_fst _ _).symm)


theorem t'_jii (i j : D.J) : D.t' j i i = pullback.fst _ _ ≫ D.t j i ≫ inv (pullback.snd _ _) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    D : CategoryTheory.GlueData C
    i j : D.J
    ⊢ Eq (D.t' j i i) (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.p …
  -/
  rw [← Category.assoc, ← D.t_fac]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    D : CategoryTheory.GlueData C
    i j : D.J
    ⊢ Eq (D.t' j i i) (CategoryTheory.CategoryStruct.comp (CategoryTheory.Category …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem t'_iji (i j : D.J) : D.t' i j i = pullback.fst _ _ ≫ D.t i j ≫ inv (pullback.snd _ _) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    D : CategoryTheory.GlueData C
    i j : D.J
    ⊢ Eq (D.t' i j i) (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.p …
  -/
  rw [← Category.assoc, ← D.t_fac]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    D : CategoryTheory.GlueData C
    i j : D.J
    ⊢ Eq (D.t' i j i) (CategoryTheory.CategoryStruct.comp (CategoryTheory.Category …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc, elementwise (attr := simp)]
theorem t_inv (i j : D.J) : D.t i j ≫ D.t j i = 𝟙 _ := by
  have eq : (pullbackSymmetry (D.f i i) (D.f i j)).hom =
      pullback.snd _ _ ≫ inv (pullback.fst _ _) := by simp
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    D : CategoryTheory.GlueData C
    i j : D.J
    eq : Eq (CategoryTheory.Limits.pullbackSymmetry (D.f i i) (D.f i j)).hom (Cate …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.t i j) (D.t j i)) (CategoryTheory. …
  -/
  have := D.cocycle i j i
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    D : CategoryTheory.GlueData C
    i j : D.J
    eq : Eq (CategoryTheory.Limits.pullbackSymmetry (D.f i i) (D.f i j)).hom (Cate …
    this : Eq (CategoryTheory.CategoryStruct.comp (D.t' i j i) (CategoryTheory.Cat …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.t i j) (D.t j i)) (CategoryTheory. …
  -/
  rw [D.t'_iij, D.t'_jii, D.t'_iji, fst_eq_snd_of_mono_eq, eq] at this
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    D : CategoryTheory.GlueData C
    i j : D.J
    eq : Eq (CategoryTheory.Limits.pullbackSymmetry (D.f i i) (D.f i j)).hom (Cate …
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.t i j) (D.t j i)) (CategoryTheory. …
  -/
  simp only [Category.assoc, IsIso.inv_hom_id_assoc] at this
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    D : CategoryTheory.GlueData C
    i j : D.J
    eq : Eq (CategoryTheory.Limits.pullbackSymmetry (D.f i i) (D.f i j)).hom (Cate …
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback. …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.t i j) (D.t j i)) (CategoryTheory. …
  -/
  rw [← IsIso.eq_inv_comp, ← Category.assoc, IsIso.comp_inv_eq] at this
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    D : CategoryTheory.GlueData C
    i j : D.J
    eq : Eq (CategoryTheory.Limits.pullbackSymmetry (D.f i i) (D.f i j)).hom (Cate …
    this : Eq (CategoryTheory.CategoryStruct.comp (D.t i j) (D.t j i)) (CategoryTh …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.t i j) (D.t j i)) (CategoryTheory. …
  -/
  simpa using this
  /-
    🎉 no goals
  -/


theorem t'_inv (i j k : D.J) :
    D.t' i j k ≫ (pullbackSymmetry _ _).hom ≫ D.t' j i k ≫ (pullbackSymmetry _ _).hom = 𝟙 _ := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    D : CategoryTheory.GlueData C
    i j k : D.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.t' i j k) (CategoryTheory.Category …
  -/
  rw [← cancel_mono (pullback.fst (D.f i j) (D.f i k))]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    D : CategoryTheory.GlueData C
    i j k : D.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp [t_fac, t_fac_assoc]
  /-
    🎉 no goals
  -/


instance t_isIso (i j : D.J) : IsIso (D.t i j) :=
  ⟨⟨D.t j i, D.t_inv _ _, D.t_inv _ _⟩⟩


instance t'_isIso (i j k : D.J) : IsIso (D.t' i j k) :=
                                                 /-
                                                   C : Type u₁
                                                   inst✝¹ : CategoryTheory.Category.{v, u₁} C
                                                   C' : Type u₂
                                                   inst✝ : CategoryTheory.Category.{v, u₂} C'
                                                   D : CategoryTheory.GlueData C
                                                   i j k : D.J
                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                                                 -/
  ⟨⟨D.t' j k i ≫ D.t' k i j, D.cocycle _ _ _, by simpa using D.cocycle _ _ _⟩⟩
                                                 /-
                                                   🎉 no goals
                                                 -/


@[reassoc]
theorem t'_comp_eq_pullbackSymmetry (i j k : D.J) :
    D.t' j k i ≫ D.t' k i j =
      (pullbackSymmetry _ _).hom ≫ D.t' j i k ≫ (pullbackSymmetry _ _).hom := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    D : CategoryTheory.GlueData C
    i j k : D.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.t' j k i) (D.t' k i j)) (CategoryT …
  -/
  trans inv (D.t' i j k)
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v, u₁} C
      D : CategoryTheory.GlueData C
      i j k : D.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.t' j k i) (D.t' k i j)) (CategoryT …
    -/
  · exact IsIso.eq_inv_of_hom_inv_id (D.cocycle _ _ _)
    /-
      🎉 no goals
    -/
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v, u₁} C
      D : CategoryTheory.GlueData C
      i j k : D.J
      ⊢ Eq (CategoryTheory.inv (D.t' i j k)) (CategoryTheory.CategoryStruct.comp (Ca …
    -/
  · rw [← cancel_mono (pullback.fst (D.f i j) (D.f i k))]
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v, u₁} C
      D : CategoryTheory.GlueData C
      i j k : D.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv (D.t' i j k)) (Ca …
    -/
    simp [t_fac, t_fac_assoc]
    /-
      🎉 no goals
    -/


/-- (Implementation) The disjoint union of `U i`. -/
def sigmaOpens [HasCoproduct D.U] : C :=
  ∐ D.U


/-- (Implementation) The diagram to take colimit of. -/
def diagram : MultispanIndex C where
  L := D.J × D.J
  R := D.J
  fstFrom := _root_.Prod.fst
  sndFrom := _root_.Prod.snd
  left := D.V
  right := D.U
  fst := fun ⟨i, j⟩ => D.f i j
  snd := fun ⟨i, j⟩ => D.t i j ≫ D.f j i


@[simp]
theorem diagram_l : D.diagram.L = (D.J × D.J) :=
  rfl


@[simp]
theorem diagram_r : D.diagram.R = D.J :=
  rfl


@[simp]
theorem diagram_fstFrom (i j : D.J) : D.diagram.fstFrom ⟨i, j⟩ = i :=
  rfl


@[simp]
theorem diagram_sndFrom (i j : D.J) : D.diagram.sndFrom ⟨i, j⟩ = j :=
  rfl


@[simp]
theorem diagram_fst (i j : D.J) : D.diagram.fst ⟨i, j⟩ = D.f i j :=
  rfl


@[simp]
theorem diagram_snd (i j : D.J) : D.diagram.snd ⟨i, j⟩ = D.t i j ≫ D.f j i :=
  rfl


@[simp]
theorem diagram_left : D.diagram.left = D.V :=
  rfl


@[simp]
theorem diagram_right : D.diagram.right = D.U :=
  rfl


/-- The glued object given a family of gluing data. -/
def glued : C :=
  multicoequalizer D.diagram


/-- The map `D.U i ⟶ D.glued` for each `i`. -/
def ι (i : D.J) : D.U i ⟶ D.glued :=
  Multicoequalizer.π D.diagram i


@[elementwise (attr := simp)]
theorem glue_condition (i j : D.J) : D.t i j ≫ D.f j i ≫ D.ι j = D.f i j ≫ D.ι i :=
  (Category.assoc _ _ _).symm.trans (Multicoequalizer.condition D.diagram ⟨i, j⟩).symm


/-- The pullback cone spanned by `V i j ⟶ U i` and `V i j ⟶ U j`.
This will often be a pullback diagram. -/
def vPullbackCone (i j : D.J) : PullbackCone (D.ι i) (D.ι j) :=
                                                    /-
                                                      C : Type u₁
                                                      inst✝² : CategoryTheory.Category.{v, u₁} C
                                                      C' : Type u₂
                                                      inst✝¹ : CategoryTheory.Category.{v, u₂} C'
                                                      D : CategoryTheory.GlueData C
                                                      inst✝ : CategoryTheory.Limits.HasMulticoequalizer D.diagram
                                                      i j : D.J
                                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.f i j) (D.ι i)) (CategoryTheory.Ca …
                                                    -/
  PullbackCone.mk (D.f i j) (D.t i j ≫ D.f j i) (by simp)
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- The projection `∐ D.U ⟶ D.glued` given by the colimit. -/
def π : D.sigmaOpens ⟶ D.glued :=
  Multicoequalizer.sigmaπ D.diagram


instance π_epi : Epi D.π := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v, u₁} C
    C' : Type u₂
    inst✝² : CategoryTheory.Category.{v, u₂} C'
    D : CategoryTheory.GlueData C
    inst✝¹ : CategoryTheory.Limits.HasMulticoequalizer D.diagram
    inst✝ : CategoryTheory.Limits.HasColimits C
    ⊢ CategoryTheory.Epi D.π
  -/
  unfold π
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v, u₁} C
    C' : Type u₂
    inst✝² : CategoryTheory.Category.{v, u₂} C'
    D : CategoryTheory.GlueData C
    inst✝¹ : CategoryTheory.Limits.HasMulticoequalizer D.diagram
    inst✝ : CategoryTheory.Limits.HasColimits C
    ⊢ CategoryTheory.Epi (CategoryTheory.Limits.Multicoequalizer.sigmaπ D.diagram)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem types_π_surjective (D : GlueData Type*) : Function.Surjective D.π :=
  (epi_iff_surjective _).mp inferInstance


theorem types_ι_jointly_surjective (D : GlueData (Type v)) (x : D.glued) :
    ∃ (i : _) (y : D.U i), D.ι i y = x := by
  /-
    D : CategoryTheory.GlueData (Type v)
    x : D.glued
    ⊢ Exists fun i => Exists fun y => Eq (D.ι i y) x
  -/
  delta CategoryTheory.GlueData.ι
  /-
    D : CategoryTheory.GlueData (Type v)
    x : D.glued
    ⊢ Exists fun i => Exists fun y => Eq (CategoryTheory.Limits.Multicoequalizer.π …
  -/
  simp_rw [← Multicoequalizer.ι_sigmaπ D.diagram]
  /-
    D : CategoryTheory.GlueData (Type v)
    x : D.glued
    ⊢ Exists fun i => Exists fun y => Eq (CategoryTheory.CategoryStruct.comp (Cate …
  -/
  rcases D.types_π_surjective x with ⟨x', rfl⟩
  --have := colimit.isoColimitCocone (Types.coproductColimitCocone _)
  rw [← show (colimit.isoColimitCocone (Types.coproductColimitCocone.{v, v} _)).inv _ = x' from
      ConcreteCategory.congr_hom
        (colimit.isoColimitCocone (Types.coproductColimitCocone _)).hom_inv_id x']
  /-
    case intro
    D : CategoryTheory.GlueData (Type v)
    x' : D.sigmaOpens
    ⊢ Exists fun i => Exists fun y => Eq (CategoryTheory.CategoryStruct.comp (Cate …
  -/
  rcases (colimit.isoColimitCocone (Types.coproductColimitCocone _)).hom x' with ⟨i, y⟩
  exact ⟨i, y, by
    simp [← Multicoequalizer.ι_sigmaπ]
    rfl ⟩


instance (i j k : D.J) : HasPullback (F.map (D.f i j)) (F.map (D.f i k)) :=
  ⟨⟨⟨_, isLimitOfHasPullbackOfPreservesLimit F (D.f i j) (D.f i k)⟩⟩⟩


/-- A functor that preserves the pullbacks of `f i j` and `f i k` can map a family of glue data. -/
@[simps]
def mapGlueData : GlueData C' where
  J := D.J
  U i := F.obj (D.U i)
  V i := F.obj (D.V i)
  f i j := F.map (D.f i j)
  f_mono _ _ := preserves_mono_of_preservesLimit _ _
  f_id _ := inferInstance
  t i j := F.map (D.t i j)
  t_id i := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v, u₁} C
      C' : Type u₂
      inst✝¹ : CategoryTheory.Category.{v, u₂} C'
      D : CategoryTheory.GlueData C
      F : CategoryTheory.Functor C C'
      inst✝ : ∀ (i j k : D.J), CategoryTheory.Limits.PreservesLimit (CategoryTheory. …
      i : D.J
      ⊢ Eq ((fun i j => F.map (D.t i j)) i i) (CategoryTheory.CategoryStruct.id ((fu …
    -/
    simp [D.t_id i]
    /-
      🎉 no goals
    -/
  t' i j k :=
    (PreservesPullback.iso F (D.f i j) (D.f i k)).inv ≫
      F.map (D.t' i j k) ≫ (PreservesPullback.iso F (D.f j k) (D.f j i)).hom
                    /-
                      C : Type u₁
                      inst✝² : CategoryTheory.Category.{v, u₁} C
                      C' : Type u₂
                      inst✝¹ : CategoryTheory.Category.{v, u₂} C'
                      D : CategoryTheory.GlueData C
                      F : CategoryTheory.Functor C C'
                      inst✝ : ∀ (i j k : D.J), CategoryTheory.Limits.PreservesLimit (CategoryTheory. …
                      i j k : D.J
                      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i j k => CategoryTheory.Categor …
                    -/
  t_fac i j k := by simpa [Iso.inv_comp_eq] using congr_arg (fun f => F.map f) (D.t_fac i j k)
                    /-
                      🎉 no goals
                    -/
  cocycle i j k := by
    simp only [Category.assoc, Iso.hom_inv_id_assoc, ← Functor.map_comp_assoc, D.cocycle,
      Iso.inv_hom_id, CategoryTheory.Functor.map_id, Category.id_comp]


/-- The diagram of the image of a `GlueData` under a functor `F` is naturally isomorphic to the
original diagram of the `GlueData` via `F`.
-/
def diagramIso : D.diagram.multispan ⋙ F ≅ (D.mapGlueData F).diagram.multispan :=
  NatIso.ofComponents
    (fun x =>
      match x with
      | WalkingMultispan.left _ => Iso.refl _
      | WalkingMultispan.right _ => Iso.refl _)
    (by
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v, u₁} C
        C' : Type u₂
        inst✝¹ : CategoryTheory.Category.{v, u₂} C'
        D : CategoryTheory.GlueData C
        F : CategoryTheory.Functor C C'
        inst✝ : ∀ (i j k : D.J), CategoryTheory.Limits.PreservesLimit (CategoryTheory. …
        ⊢ ∀ {X Y : CategoryTheory.Limits.WalkingMultispan D.diagram.fstFrom D.diagram. …
      -/
      rintro (⟨_, _⟩ | _) _ (_ | _ | _)
        /-
          case left.mk.id
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v, u₁} C
          C' : Type u₂
          inst✝¹ : CategoryTheory.Category.{v, u₂} C'
          D : CategoryTheory.GlueData C
          F : CategoryTheory.Functor C C'
          inst✝ : ∀ (i j k : D.J), CategoryTheory.Limits.PreservesLimit (CategoryTheory. …
          fst✝ snd✝ : D.J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((D.diagram.multispan.comp F).map (Ca …
        -/
      · erw [Category.comp_id, Category.id_comp, Functor.map_id]
        /-
          case left.mk.id
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v, u₁} C
          C' : Type u₂
          inst✝¹ : CategoryTheory.Category.{v, u₂} C'
          D : CategoryTheory.GlueData C
          F : CategoryTheory.Functor C C'
          inst✝ : ∀ (i j k : D.J), CategoryTheory.Limits.PreservesLimit (CategoryTheory. …
          fst✝ snd✝ : D.J
          ⊢ Eq (CategoryTheory.CategoryStruct.id ((D.diagram.multispan.comp F).obj (Cate …
        -/
        rfl
        /-
          🎉 no goals
        -/
        /-
          case left.mk.fst
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v, u₁} C
          C' : Type u₂
          inst✝¹ : CategoryTheory.Category.{v, u₂} C'
          D : CategoryTheory.GlueData C
          F : CategoryTheory.Functor C C'
          inst✝ : ∀ (i j k : D.J), CategoryTheory.Limits.PreservesLimit (CategoryTheory. …
          fst✝ snd✝ : D.J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((D.diagram.multispan.comp F).map (Ca …
        -/
      · erw [Category.comp_id, Category.id_comp]
        /-
          case left.mk.fst
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v, u₁} C
          C' : Type u₂
          inst✝¹ : CategoryTheory.Category.{v, u₂} C'
          D : CategoryTheory.GlueData C
          F : CategoryTheory.Functor C C'
          inst✝ : ∀ (i j k : D.J), CategoryTheory.Limits.PreservesLimit (CategoryTheory. …
          fst✝ snd✝ : D.J
          ⊢ Eq ((D.diagram.multispan.comp F).map (CategoryTheory.Limits.WalkingMultispan …
        -/
        rfl
        /-
          🎉 no goals
        -/
        /-
          case left.mk.snd
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v, u₁} C
          C' : Type u₂
          inst✝¹ : CategoryTheory.Category.{v, u₂} C'
          D : CategoryTheory.GlueData C
          F : CategoryTheory.Functor C C'
          inst✝ : ∀ (i j k : D.J), CategoryTheory.Limits.PreservesLimit (CategoryTheory. …
          fst✝ snd✝ : D.J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((D.diagram.multispan.comp F).map (Ca …
        -/
      · erw [Category.comp_id, Category.id_comp, Functor.map_comp]
        /-
          case left.mk.snd
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v, u₁} C
          C' : Type u₂
          inst✝¹ : CategoryTheory.Category.{v, u₂} C'
          D : CategoryTheory.GlueData C
          F : CategoryTheory.Functor C C'
          inst✝ : ∀ (i j k : D.J), CategoryTheory.Limits.PreservesLimit (CategoryTheory. …
          fst✝ snd✝ : D.J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (D.t fst✝ snd✝)) (F.map (D.f s …
        -/
        rfl
        /-
          🎉 no goals
        -/
        /-
          case right.id
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v, u₁} C
          C' : Type u₂
          inst✝¹ : CategoryTheory.Category.{v, u₂} C'
          D : CategoryTheory.GlueData C
          F : CategoryTheory.Functor C C'
          inst✝ : ∀ (i j k : D.J), CategoryTheory.Limits.PreservesLimit (CategoryTheory. …
          a✝ : D.diagram.R
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((D.diagram.multispan.comp F).map (Ca …
        -/
      · erw [Category.comp_id, Category.id_comp, Functor.map_id]
        /-
          case right.id
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v, u₁} C
          C' : Type u₂
          inst✝¹ : CategoryTheory.Category.{v, u₂} C'
          D : CategoryTheory.GlueData C
          F : CategoryTheory.Functor C C'
          inst✝ : ∀ (i j k : D.J), CategoryTheory.Limits.PreservesLimit (CategoryTheory. …
          a✝ : D.diagram.R
          ⊢ Eq (CategoryTheory.CategoryStruct.id ((D.diagram.multispan.comp F).obj (Cate …
        -/
        rfl)
        /-
          🎉 no goals
        -/


@[simp]
theorem diagramIso_app_left (i : D.J × D.J) :
    (D.diagramIso F).app (WalkingMultispan.left i) = Iso.refl _ :=
  rfl


@[simp]
theorem diagramIso_app_right (i : D.J) :
    (D.diagramIso F).app (WalkingMultispan.right i) = Iso.refl _ :=
  rfl


@[simp]
theorem diagramIso_hom_app_left (i : D.J × D.J) :
    (D.diagramIso F).hom.app (WalkingMultispan.left i) = 𝟙 _ :=
  rfl


@[simp]
theorem diagramIso_hom_app_right (i : D.J) :
    (D.diagramIso F).hom.app (WalkingMultispan.right i) = 𝟙 _ :=
  rfl


@[simp]
theorem diagramIso_inv_app_left (i : D.J × D.J) :
    (D.diagramIso F).inv.app (WalkingMultispan.left i) = 𝟙 _ :=
  rfl


@[simp]
theorem diagramIso_inv_app_right (i : D.J) :
    (D.diagramIso F).inv.app (WalkingMultispan.right i) = 𝟙 _ :=
  rfl


theorem hasColimit_multispan_comp : HasColimit (D.diagram.multispan ⋙ F) :=
  ⟨⟨⟨_, isColimitOfPreserves _ (colimit.isColimit _)⟩⟩⟩


theorem hasColimit_mapGlueData_diagram : HasMulticoequalizer (D.mapGlueData F).diagram :=
  hasColimitOfIso (D.diagramIso F).symm


/-- If `F` preserves the gluing, we obtain an iso between the glued objects. -/
def gluedIso : F.obj D.glued ≅ (D.mapGlueData F).glued :=
  haveI : HasColimit (MultispanIndex.multispan (diagram (mapGlueData D F))) := inferInstance
  preservesColimitIso F D.diagram.multispan ≪≫ Limits.HasColimit.isoOfNatIso (D.diagramIso F)


@[reassoc (attr := simp)]
theorem ι_gluedIso_hom (i : D.J) : F.map (D.ι i) ≫ (D.gluedIso F).hom = (D.mapGlueData F).ι i := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v, u₁} C
    C' : Type u₂
    inst✝³ : CategoryTheory.Category.{v, u₂} C'
    D : CategoryTheory.GlueData C
    F : CategoryTheory.Functor C C'
    inst✝² : CategoryTheory.Limits.HasMulticoequalizer D.diagram
    inst✝¹ : CategoryTheory.Limits.PreservesColimit D.diagram.multispan F
    inst✝ : ∀ (i j k : D.J), CategoryTheory.Limits.PreservesLimit (CategoryTheory. …
    i : D.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (D.ι i)) (D.gluedIso F).hom) ( …
  -/
  haveI : HasColimit (MultispanIndex.multispan (diagram (mapGlueData D F))) := inferInstance
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v, u₁} C
    C' : Type u₂
    inst✝³ : CategoryTheory.Category.{v, u₂} C'
    D : CategoryTheory.GlueData C
    F : CategoryTheory.Functor C C'
    inst✝² : CategoryTheory.Limits.HasMulticoequalizer D.diagram
    inst✝¹ : CategoryTheory.Limits.PreservesColimit D.diagram.multispan F
    inst✝ : ∀ (i j k : D.J), CategoryTheory.Limits.PreservesLimit (CategoryTheory. …
    i : D.J
    this : CategoryTheory.Limits.HasColimit (D.mapGlueData F).diagram.multispan
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (D.ι i)) (D.gluedIso F).hom) ( …
  -/
  erw [ι_preservesColimitIso_hom_assoc]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v, u₁} C
    C' : Type u₂
    inst✝³ : CategoryTheory.Category.{v, u₂} C'
    D : CategoryTheory.GlueData C
    F : CategoryTheory.Functor C C'
    inst✝² : CategoryTheory.Limits.HasMulticoequalizer D.diagram
    inst✝¹ : CategoryTheory.Limits.PreservesColimit D.diagram.multispan F
    inst✝ : ∀ (i j k : D.J), CategoryTheory.Limits.PreservesLimit (CategoryTheory. …
    i : D.J
    this : CategoryTheory.Limits.HasColimit (D.mapGlueData F).diagram.multispan
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (D.d …
  -/
  rw [HasColimit.isoOfNatIso_ι_hom]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v, u₁} C
    C' : Type u₂
    inst✝³ : CategoryTheory.Category.{v, u₂} C'
    D : CategoryTheory.GlueData C
    F : CategoryTheory.Functor C C'
    inst✝² : CategoryTheory.Limits.HasMulticoequalizer D.diagram
    inst✝¹ : CategoryTheory.Limits.PreservesColimit D.diagram.multispan F
    inst✝ : ∀ (i j k : D.J), CategoryTheory.Limits.PreservesLimit (CategoryTheory. …
    i : D.J
    this : CategoryTheory.Limits.HasColimit (D.mapGlueData F).diagram.multispan
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((D.diagramIso F).hom.app (CategoryTh …
  -/
  erw [Category.id_comp]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v, u₁} C
    C' : Type u₂
    inst✝³ : CategoryTheory.Category.{v, u₂} C'
    D : CategoryTheory.GlueData C
    F : CategoryTheory.Functor C C'
    inst✝² : CategoryTheory.Limits.HasMulticoequalizer D.diagram
    inst✝¹ : CategoryTheory.Limits.PreservesColimit D.diagram.multispan F
    inst✝ : ∀ (i j k : D.J), CategoryTheory.Limits.PreservesLimit (CategoryTheory. …
    i : D.J
    this : CategoryTheory.Limits.HasColimit (D.mapGlueData F).diagram.multispan
    ⊢ Eq (CategoryTheory.Limits.colimit.ι (D.mapGlueData F).diagram.multispan (Cat …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem ι_gluedIso_inv (i : D.J) : (D.mapGlueData F).ι i ≫ (D.gluedIso F).inv = F.map (D.ι i) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v, u₁} C
    C' : Type u₂
    inst✝³ : CategoryTheory.Category.{v, u₂} C'
    D : CategoryTheory.GlueData C
    F : CategoryTheory.Functor C C'
    inst✝² : CategoryTheory.Limits.HasMulticoequalizer D.diagram
    inst✝¹ : CategoryTheory.Limits.PreservesColimit D.diagram.multispan F
    inst✝ : ∀ (i j k : D.J), CategoryTheory.Limits.PreservesLimit (CategoryTheory. …
    i : D.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((D.mapGlueData F).ι i) (D.gluedIso F …
  -/
  rw [Iso.comp_inv_eq, ι_gluedIso_hom]
  /-
    🎉 no goals
  -/


/-- If `F` preserves the gluing, and reflects the pullback of `U i ⟶ glued` and `U j ⟶ glued`,
then `F` reflects the fact that `V_pullback_cone` is a pullback. -/
def vPullbackConeIsLimitOfMap (i j : D.J) [ReflectsLimit (cospan (D.ι i) (D.ι j)) F]
    (hc : IsLimit ((D.mapGlueData F).vPullbackCone i j)) : IsLimit (D.vPullbackCone i j) := by
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v, u₁} C
    C' : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v, u₂} C'
    D : CategoryTheory.GlueData C
    F : CategoryTheory.Functor C C'
    inst✝³ : CategoryTheory.Limits.HasMulticoequalizer D.diagram
    inst✝² : CategoryTheory.Limits.PreservesColimit D.diagram.multispan F
    inst✝¹ : ∀ (i j k : D.J), CategoryTheory.Limits.PreservesLimit (CategoryTheory …
    i j : D.J
    inst✝ : CategoryTheory.Limits.ReflectsLimit (CategoryTheory.Limits.cospan (D.ι …
    hc : CategoryTheory.Limits.IsLimit ((D.mapGlueData F).vPullbackCone i j)
    ⊢ CategoryTheory.Limits.IsLimit (D.vPullbackCone i j)
  -/
  apply isLimitOfReflects F
  /-
    case t
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v, u₁} C
    C' : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v, u₂} C'
    D : CategoryTheory.GlueData C
    F : CategoryTheory.Functor C C'
    inst✝³ : CategoryTheory.Limits.HasMulticoequalizer D.diagram
    inst✝² : CategoryTheory.Limits.PreservesColimit D.diagram.multispan F
    inst✝¹ : ∀ (i j k : D.J), CategoryTheory.Limits.PreservesLimit (CategoryTheory …
    i j : D.J
    inst✝ : CategoryTheory.Limits.ReflectsLimit (CategoryTheory.Limits.cospan (D.ι …
    hc : CategoryTheory.Limits.IsLimit ((D.mapGlueData F).vPullbackCone i j)
    ⊢ CategoryTheory.Limits.IsLimit (F.mapCone (D.vPullbackCone i j))
  -/
  apply (isLimitMapConePullbackConeEquiv _ _).symm _
  let e : cospan (F.map (D.ι i)) (F.map (D.ι j)) ≅
      cospan ((D.mapGlueData F).ι i) ((D.mapGlueData F).ι j) :=
    NatIso.ofComponents
      (fun x => by
        cases x
        exacts [D.gluedIso F, Iso.refl _])
      (by rintro (_ | _) (_ | _) (_ | _ | _) <;> simp)
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v, u₁} C
    C' : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v, u₂} C'
    D : CategoryTheory.GlueData C
    F : CategoryTheory.Functor C C'
    inst✝³ : CategoryTheory.Limits.HasMulticoequalizer D.diagram
    inst✝² : CategoryTheory.Limits.PreservesColimit D.diagram.multispan F
    inst✝¹ : ∀ (i j k : D.J), CategoryTheory.Limits.PreservesLimit (CategoryTheory …
    i j : D.J
    inst✝ : CategoryTheory.Limits.ReflectsLimit (CategoryTheory.Limits.cospan (D.ι …
    hc : CategoryTheory.Limits.IsLimit ((D.mapGlueData F).vPullbackCone i j)
    e : CategoryTheory.Iso (CategoryTheory.Limits.cospan (F.map (D.ι i)) (F.map (D …
    ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk (F.map  …
  -/
  apply IsLimit.postcomposeHomEquiv e _ _
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v, u₁} C
    C' : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v, u₂} C'
    D : CategoryTheory.GlueData C
    F : CategoryTheory.Functor C C'
    inst✝³ : CategoryTheory.Limits.HasMulticoequalizer D.diagram
    inst✝² : CategoryTheory.Limits.PreservesColimit D.diagram.multispan F
    inst✝¹ : ∀ (i j k : D.J), CategoryTheory.Limits.PreservesLimit (CategoryTheory …
    i j : D.J
    inst✝ : CategoryTheory.Limits.ReflectsLimit (CategoryTheory.Limits.cospan (D.ι …
    hc : CategoryTheory.Limits.IsLimit ((D.mapGlueData F).vPullbackCone i j)
    e : CategoryTheory.Iso (CategoryTheory.Limits.cospan (F.map (D.ι i)) (F.map (D …
    ⊢ CategoryTheory.Limits.IsLimit ((CategoryTheory.Limits.Cones.postcompose e.ho …
  -/
  apply hc.ofIsoLimit
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v, u₁} C
    C' : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v, u₂} C'
    D : CategoryTheory.GlueData C
    F : CategoryTheory.Functor C C'
    inst✝³ : CategoryTheory.Limits.HasMulticoequalizer D.diagram
    inst✝² : CategoryTheory.Limits.PreservesColimit D.diagram.multispan F
    inst✝¹ : ∀ (i j k : D.J), CategoryTheory.Limits.PreservesLimit (CategoryTheory …
    i j : D.J
    inst✝ : CategoryTheory.Limits.ReflectsLimit (CategoryTheory.Limits.cospan (D.ι …
    hc : CategoryTheory.Limits.IsLimit ((D.mapGlueData F).vPullbackCone i j)
    e : CategoryTheory.Iso (CategoryTheory.Limits.cospan (F.map (D.ι i)) (F.map (D …
    ⊢ CategoryTheory.Iso ((D.mapGlueData F).vPullbackCone i j) ((CategoryTheory.Li …
  -/
  refine Cones.ext (Iso.refl _) ?_
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v, u₁} C
    C' : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v, u₂} C'
    D : CategoryTheory.GlueData C
    F : CategoryTheory.Functor C C'
    inst✝³ : CategoryTheory.Limits.HasMulticoequalizer D.diagram
    inst✝² : CategoryTheory.Limits.PreservesColimit D.diagram.multispan F
    inst✝¹ : ∀ (i j k : D.J), CategoryTheory.Limits.PreservesLimit (CategoryTheory …
    i j : D.J
    inst✝ : CategoryTheory.Limits.ReflectsLimit (CategoryTheory.Limits.cospan (D.ι …
    hc : CategoryTheory.Limits.IsLimit ((D.mapGlueData F).vPullbackCone i j)
    e : CategoryTheory.Iso (CategoryTheory.Limits.cospan (F.map (D.ι i)) (F.map (D …
    ⊢ ∀ (j_1 : CategoryTheory.Limits.WalkingCospan), Eq (((D.mapGlueData F).vPullb …
  -/
  rintro (_ | _ | _)
  /-
    case none
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v, u₁} C
    C' : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v, u₂} C'
    D : CategoryTheory.GlueData C
    F : CategoryTheory.Functor C C'
    inst✝³ : CategoryTheory.Limits.HasMulticoequalizer D.diagram
    inst✝² : CategoryTheory.Limits.PreservesColimit D.diagram.multispan F
    inst✝¹ : ∀ (i j k : D.J), CategoryTheory.Limits.PreservesLimit (CategoryTheory …
    i j : D.J
    inst✝ : CategoryTheory.Limits.ReflectsLimit (CategoryTheory.Limits.cospan (D.ι …
    hc : CategoryTheory.Limits.IsLimit ((D.mapGlueData F).vPullbackCone i j)
    e : CategoryTheory.Iso (CategoryTheory.Limits.cospan (F.map (D.ι i)) (F.map (D …
    ⊢ Eq (((D.mapGlueData F).vPullbackCone i j).π.app Option.none) (CategoryTheory …
  -/
  all_goals simp [e]; rfl
  /-
    🎉 no goals
  -/


/-- If there is a forgetful functor into `Type` that preserves enough (co)limits, then `D.ι` will
be jointly surjective. -/
theorem ι_jointly_surjective (F : C ⥤ Type v) [PreservesColimit D.diagram.multispan F]
    [∀ i j k : D.J, PreservesLimit (cospan (D.f i j) (D.f i k)) F] (x : F.obj D.glued) :
    ∃ (i : _) (y : F.obj (D.U i)), F.map (D.ι i) y = x := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v, u₁} C
    D : CategoryTheory.GlueData C
    inst✝² : CategoryTheory.Limits.HasMulticoequalizer D.diagram
    F : CategoryTheory.Functor C (Type v)
    inst✝¹ : CategoryTheory.Limits.PreservesColimit D.diagram.multispan F
    inst✝ : ∀ (i j k : D.J), CategoryTheory.Limits.PreservesLimit (CategoryTheory. …
    x : F.obj D.glued
    ⊢ Exists fun i => Exists fun y => Eq (F.map (D.ι i) y) x
  -/
  let e := D.gluedIso F
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v, u₁} C
    D : CategoryTheory.GlueData C
    inst✝² : CategoryTheory.Limits.HasMulticoequalizer D.diagram
    F : CategoryTheory.Functor C (Type v)
    inst✝¹ : CategoryTheory.Limits.PreservesColimit D.diagram.multispan F
    inst✝ : ∀ (i j k : D.J), CategoryTheory.Limits.PreservesLimit (CategoryTheory. …
    x : F.obj D.glued
    e : CategoryTheory.Iso (F.obj D.glued) (D.mapGlueData F).glued := D.gluedIso F
    ⊢ Exists fun i => Exists fun y => Eq (F.map (D.ι i) y) x
  -/
  obtain ⟨i, y, eq⟩ := (D.mapGlueData F).types_ι_jointly_surjective (e.hom x)
  /-
    case intro.intro
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v, u₁} C
    D : CategoryTheory.GlueData C
    inst✝² : CategoryTheory.Limits.HasMulticoequalizer D.diagram
    F : CategoryTheory.Functor C (Type v)
    inst✝¹ : CategoryTheory.Limits.PreservesColimit D.diagram.multispan F
    inst✝ : ∀ (i j k : D.J), CategoryTheory.Limits.PreservesLimit (CategoryTheory. …
    x : F.obj D.glued
    e : CategoryTheory.Iso (F.obj D.glued) (D.mapGlueData F).glued := D.gluedIso F
    i : (D.mapGlueData F).J
    y : (D.mapGlueData F).U i
    eq : Eq ((D.mapGlueData F).ι i y) (e.hom x)
    ⊢ Exists fun i => Exists fun y => Eq (F.map (D.ι i) y) x
  -/
  replace eq := congr_arg e.inv eq
  /-
    case intro.intro
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v, u₁} C
    D : CategoryTheory.GlueData C
    inst✝² : CategoryTheory.Limits.HasMulticoequalizer D.diagram
    F : CategoryTheory.Functor C (Type v)
    inst✝¹ : CategoryTheory.Limits.PreservesColimit D.diagram.multispan F
    inst✝ : ∀ (i j k : D.J), CategoryTheory.Limits.PreservesLimit (CategoryTheory. …
    x : F.obj D.glued
    e : CategoryTheory.Iso (F.obj D.glued) (D.mapGlueData F).glued := D.gluedIso F
    i : (D.mapGlueData F).J
    y : (D.mapGlueData F).U i
    eq : Eq (e.inv ((D.mapGlueData F).ι i y)) (e.inv (e.hom x))
    ⊢ Exists fun i => Exists fun y => Eq (F.map (D.ι i) y) x
  -/
  change ((D.mapGlueData F).ι i ≫ e.inv) y = (e.hom ≫ e.inv) x at eq
  /-
    case intro.intro
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v, u₁} C
    D : CategoryTheory.GlueData C
    inst✝² : CategoryTheory.Limits.HasMulticoequalizer D.diagram
    F : CategoryTheory.Functor C (Type v)
    inst✝¹ : CategoryTheory.Limits.PreservesColimit D.diagram.multispan F
    inst✝ : ∀ (i j k : D.J), CategoryTheory.Limits.PreservesLimit (CategoryTheory. …
    x : F.obj D.glued
    e : CategoryTheory.Iso (F.obj D.glued) (D.mapGlueData F).glued := D.gluedIso F
    i : (D.mapGlueData F).J
    y : (D.mapGlueData F).U i
    eq : Eq (CategoryTheory.CategoryStruct.comp ((D.mapGlueData F).ι i) e.inv y) ( …
    ⊢ Exists fun i => Exists fun y => Eq (F.map (D.ι i) y) x
  -/
  rw [e.hom_inv_id, D.ι_gluedIso_inv] at eq
  /-
    case intro.intro
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v, u₁} C
    D : CategoryTheory.GlueData C
    inst✝² : CategoryTheory.Limits.HasMulticoequalizer D.diagram
    F : CategoryTheory.Functor C (Type v)
    inst✝¹ : CategoryTheory.Limits.PreservesColimit D.diagram.multispan F
    inst✝ : ∀ (i j k : D.J), CategoryTheory.Limits.PreservesLimit (CategoryTheory. …
    x : F.obj D.glued
    e : CategoryTheory.Iso (F.obj D.glued) (D.mapGlueData F).glued := D.gluedIso F
    i : (D.mapGlueData F).J
    y : (D.mapGlueData F).U i
    eq : Eq (F.map (D.ι i) y) (CategoryTheory.CategoryStruct.id (F.obj D.glued) x)
    ⊢ Exists fun i => Exists fun y => Eq (F.map (D.ι i) y) x
  -/
  exact ⟨i, y, eq⟩
  /-
    🎉 no goals
  -/


/--
This is a variant of `GlueData` that only requires conditions on `V (i, j)` when `i ≠ j`.
See `GlueData.ofGlueData'`
-/
structure GlueData' where
  /-- Indexing type of a glue data. -/
  J : Type v
  /-- Objects of a glue data to be glued. -/
  U : J → C
  /-- Objects representing the intersections. -/
  V : ∀ (i j : J), i ≠ j → C
  /-- The inclusion maps of the intersection into the object. -/
  f : ∀ i j h, V i j h ⟶ U i
  f_mono : ∀ i j h, Mono (f i j h) := by infer_instance
  f_hasPullback : ∀ i j k hij hik, HasPullback (f i j hij) (f i k hik) := by infer_instance
  /-- The transition maps between the intersections. -/
  t : ∀ i j h, V i j h ⟶ V j i h.symm
  /-- The transition maps between the intersection of intersections. -/
  t' : ∀ i j k hij hik hjk,
    pullback (f i j hij) (f i k hik) ⟶ pullback (f j k hjk) (f j i hij.symm)
  t_fac : ∀ i j k hij hik hjk, t' i j k hij hik hjk ≫ pullback.snd _ _ =
    pullback.fst _ _ ≫ t i j hij
  t_inv : ∀ i j hij, t i j hij ≫ t j i hij.symm = 𝟙 _
  cocycle : ∀ i j k hij hik hjk, t' i j k hij hik hjk ≫
    t' j k i hjk hij.symm hik.symm ≫ t' k i j hik.symm hjk.symm hij = 𝟙 _


attribute [reassoc (attr := simp)] GlueData'.t_inv GlueData'.cocycle


open scoped Classical in
/-- (Implementation detail) the constructed `GlueData.f` from a `GlueData'`. -/
abbrev GlueData'.f' (D : GlueData' C) (i j : D.J) :
    (if h : i = j then D.U i else D.V i j h) ⟶ D.U i :=
  if h : i = j then eqToHom (dif_pos h) else eqToHom (dif_neg h) ≫ D.f i j h


instance (D : GlueData' C) (i j : D.J) :
                          /-
                            C : Type u₁
                            inst✝¹ : CategoryTheory.Category.{v, u₁} C
                            C' : Type u₂
                            inst✝ : CategoryTheory.Category.{v, u₂} C'
                            D : CategoryTheory.GlueData' C
                            i j : D.J
                            ⊢ CategoryTheory.Mono (D.f' i j)
                          -/
                                                              /-
                                                                🎉 no goals
                                                              -/
    Mono (D.f' i j) := by dsimp [GlueData'.f']; split_ifs <;> infer_instance
                                                              /-
                                                                🎉 no goals
                                                              -/


instance (D : GlueData' C) (i : D.J) :
                           /-
                             C : Type u₁
                             inst✝¹ : CategoryTheory.Category.{v, u₁} C
                             C' : Type u₂
                             inst✝ : CategoryTheory.Category.{v, u₂} C'
                             D : CategoryTheory.GlueData' C
                             i : D.J
                             ⊢ CategoryTheory.IsIso (D.f' i i)
                           -/
    IsIso (D.f' i i) := by simp only [GlueData'.f', ↓reduceDIte]; infer_instance
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


instance (D : GlueData' C) (i j k : D.J) :
    HasPullback (D.f' i j) (D.f' i k) := by
  if hij : i = j then
    apply (config := { allowSynthFailures := true}) hasPullback_of_left_iso
    simp only [GlueData'.f', dif_pos hij]
    infer_instance
  else if hik : i = k then
    apply (config := { allowSynthFailures := true}) hasPullback_of_right_iso
    simp only [GlueData'.f', dif_pos hik]
    infer_instance
  else
    have {X Y Z : C} (f : X ⟶ Y) (e : Z = X) : HEq (eqToHom e ≫ f) f := by subst e; simp
    convert D.f_hasPullback i j k hij hik <;> simp [GlueData'.f', hij, hik, this]


open scoped Classical in
/-- (Implementation detail) the constructed `GlueData.t'` from a `GlueData'`. -/
def GlueData'.t'' (D : GlueData' C) (i j k : D.J) :
    pullback (D.f' i j) (D.f' i k) ⟶ pullback (D.f' j k) (D.f' j i) :=
  if hij : i = j then
    (pullbackSymmetry _ _).hom ≫
                                        /-
                                          C : Type u₁
                                          inst✝¹ : CategoryTheory.Category.{v, u₁} C
                                          C' : Type u₂
                                          inst✝ : CategoryTheory.Category.{v, u₂} C'
                                          D : CategoryTheory.GlueData' C
                                          i j k : D.J
                                          hij : Eq i j
                                          ⊢ Eq (dite (Eq i k) (fun h => D.U i) fun h => D.V i k h) (dite (Eq j k) (fun h …
                                        -/
                                        /-
                                          🎉 no goals
                                        -/
                                                             /-
                                                               🎉 no goals
                                                             -/
      pullback.map _ _ _ _ (eqToHom (by aesop)) (eqToHom (by aesop)) (eqToHom (by aesop))
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
            /-
              C : Type u₁
              inst✝¹ : CategoryTheory.Category.{v, u₁} C
              C' : Type u₂
              inst✝ : CategoryTheory.Category.{v, u₂} C'
              D : CategoryTheory.GlueData' C
              i j k : D.J
              hij : Eq i j
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.f' i k) (CategoryTheory.eqToHom ⋯) …
            -/
            /-
              🎉 no goals
            -/
        (by aesop) (by aesop)
                       /-
                         🎉 no goals
                       -/
  else if hik : i = k then
    have : IsIso (pullback.snd (D.f' j k) (D.f' j i)) := by
      /-
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v, u₁} C
        C' : Type u₂
        inst✝ : CategoryTheory.Category.{v, u₂} C'
        D : CategoryTheory.GlueData' C
        i j k : D.J
        hij : Not (Eq i j)
        hik : Eq i k
        ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.pullback.snd (D.f' j k) (D.f' j  …
      -/
      subst hik; infer_instance
                 /-
                   🎉 no goals
                 -/
    pullback.fst _ _ ≫ eqToHom (dif_neg hij) ≫ D.t _ _ _ ≫
      eqToHom (dif_neg (Ne.symm hij)).symm ≫ inv (pullback.snd _ _)
  else if hjk : j = k then
    have : IsIso (pullback.snd (D.f' j k) (D.f' j i)) := by
      /-
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v, u₁} C
        C' : Type u₂
        inst✝ : CategoryTheory.Category.{v, u₂} C'
        D : CategoryTheory.GlueData' C
        i j k : D.J
        hij : Not (Eq i j)
        hik : Not (Eq i k)
        hjk : Eq j k
        ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.pullback.snd (D.f' j k) (D.f' j  …
      -/
      apply (config := { allowSynthFailures := true}) pullback_snd_iso_of_left_iso
      /-
        case inst
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v, u₁} C
        C' : Type u₂
        inst✝ : CategoryTheory.Category.{v, u₂} C'
        D : CategoryTheory.GlueData' C
        i j k : D.J
        hij : Not (Eq i j)
        hik : Not (Eq i k)
        hjk : Eq j k
        ⊢ CategoryTheory.IsIso (D.f' j k)
      -/
      simp only [hjk, GlueData'.f', ↓reduceDIte]
      /-
        case inst
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v, u₁} C
        C' : Type u₂
        inst✝ : CategoryTheory.Category.{v, u₂} C'
        D : CategoryTheory.GlueData' C
        i j k : D.J
        hij : Not (Eq i j)
        hik : Not (Eq i k)
        hjk : Eq j k
        ⊢ CategoryTheory.IsIso (CategoryTheory.eqToHom ⋯)
      -/
      infer_instance
      /-
        🎉 no goals
      -/
    pullback.fst _ _ ≫ eqToHom (dif_neg hij) ≫ D.t _ _ _ ≫
      eqToHom (dif_neg (Ne.symm hij)).symm ≫ inv (pullback.snd _ _)
  else
    haveI := Ne.symm hij
                                      /-
                                        C : Type u₁
                                        inst✝¹ : CategoryTheory.Category.{v, u₁} C
                                        C' : Type u₂
                                        inst✝ : CategoryTheory.Category.{v, u₂} C'
                                        D : CategoryTheory.GlueData' C
                                        i j k : D.J
                                        hij : Not (Eq i j)
                                        hik : Not (Eq i k)
                                        hjk : Not (Eq j k)
                                        this : Ne j i
                                        ⊢ Eq (dite (Eq i j) (fun h => D.U i) fun h => D.V i j h) (D.V i j hij)
                                      -/
                                      /-
                                        🎉 no goals
                                      -/
    pullback.map _ _ _ _ (eqToHom (by aesop)) (eqToHom (by rw [dif_neg hik]))
                                                           /-
                                                             🎉 no goals
                                                           -/
                     /-
                       C : Type u₁
                       inst✝¹ : CategoryTheory.Category.{v, u₁} C
                       C' : Type u₂
                       inst✝ : CategoryTheory.Category.{v, u₂} C'
                       D : CategoryTheory.GlueData' C
                       i j k : D.J
                       hij : Not (Eq i j)
                       hik : Not (Eq i k)
                       hjk : Not (Eq j k)
                       this : Ne j i
                       ⊢ Eq (D.U i) (D.U i)
                     -/
                     /-
                       🎉 no goals
                     -/
                                           /-
                                             🎉 no goals
                                           -/
        (eqToHom (by aesop)) (by delta f'; aesop) (by delta f'; aesop) ≫
                                                                /-
                                                                  🎉 no goals
                                                                -/
      D.t' i j k hij hik hjk ≫
                                        /-
                                          C : Type u₁
                                          inst✝¹ : CategoryTheory.Category.{v, u₁} C
                                          C' : Type u₂
                                          inst✝ : CategoryTheory.Category.{v, u₂} C'
                                          D : CategoryTheory.GlueData' C
                                          i j k : D.J
                                          hij : Not (Eq i j)
                                          hik : Not (Eq i k)
                                          hjk : Not (Eq j k)
                                          this : Ne j i
                                          ⊢ Eq (D.V j k hjk) (dite (Eq j k) (fun h => D.U j) fun h => D.V j k h)
                                        -/
                                        /-
                                          🎉 no goals
                                        -/
                                                             /-
                                                               🎉 no goals
                                                             -/
      pullback.map _ _ _ _ (eqToHom (by aesop)) (eqToHom (by aesop)) (eqToHom (by aesop))
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
            /-
              C : Type u₁
              inst✝¹ : CategoryTheory.Category.{v, u₁} C
              C' : Type u₂
              inst✝ : CategoryTheory.Category.{v, u₂} C'
              D : CategoryTheory.GlueData' C
              i j k : D.J
              hij : Not (Eq i j)
              hik : Not (Eq i k)
              hjk : Not (Eq j k)
              this : Ne j i
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.f j k hjk) (CategoryTheory.eqToHom …
            -/
                      /-
                        🎉 no goals
                      -/
        (by delta f'; aesop) (by delta f'; aesop)
                                           /-
                                             🎉 no goals
                                           -/


open scoped Classical in
/--
The constructed `GlueData` of a `GlueData'`, where `GlueData'` is a variant of `GlueData` that only
requires conditions on `V (i, j)` when `i ≠ j`.
-/
def GlueData.ofGlueData' (D : GlueData' C) : GlueData C where
  J := D.J
  U := D.U
  V ij := if h : ij.1 = ij.2 then D.U ij.1 else D.V ij.1 ij.2 h
  f i j := D.f' i j
               /-
                 C : Type u₁
                 inst✝¹ : CategoryTheory.Category.{v, u₁} C
                 C' : Type u₂
                 inst✝ : CategoryTheory.Category.{v, u₂} C'
                 D : CategoryTheory.GlueData' C
                 i : D.J
                 ⊢ CategoryTheory.IsIso ((fun i j => D.f' i j) i i)
               -/
  f_id i := by simp only [↓reduceDIte, GlueData'.f']; infer_instance
                                                      /-
                                                        🎉 no goals
                                                      -/
                                         /-
                                           C : Type u₁
                                           inst✝¹ : CategoryTheory.Category.{v, u₁} C
                                           C' : Type u₂
                                           inst✝ : CategoryTheory.Category.{v, u₂} C'
                                           D : CategoryTheory.GlueData' C
                                           i j : D.J
                                           h : Eq i j
                                           ⊢ Eq ((fun ij => dite (Eq ij.1 ij.2) (fun h => D.U ij.1) fun h => D.V ij.1 ij. …
                                         -/
  t i j := if h : i = j then eqToHom (by simp [h]) else
                                         /-
                                           🎉 no goals
                                         -/
    eqToHom (dif_neg h) ≫ D.t i j h ≫ eqToHom (dif_neg (Ne.symm h)).symm
               /-
                 C : Type u₁
                 inst✝¹ : CategoryTheory.Category.{v, u₁} C
                 C' : Type u₂
                 inst✝ : CategoryTheory.Category.{v, u₂} C'
                 D : CategoryTheory.GlueData' C
                 i : D.J
                 ⊢ Eq ((fun i j => dite (Eq i j) (fun h => CategoryTheory.eqToHom ⋯) fun h => C …
               -/
  t_id i := by simp
               /-
                 🎉 no goals
               -/
  t' := D.t''
  t_fac i j k := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v, u₁} C
      C' : Type u₂
      inst✝ : CategoryTheory.Category.{v, u₂} C'
      D : CategoryTheory.GlueData' C
      i j k : D.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.t'' i j k) (CategoryTheory.Limits. …
    -/
    delta GlueData'.t''
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v, u₁} C
      C' : Type u₂
      inst✝ : CategoryTheory.Category.{v, u₂} C'
      D : CategoryTheory.GlueData' C
      i j k : D.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (Eq i j) (fun hij => CategoryTh …
    -/
    split_ifs
      /-
        case pos
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v, u₁} C
        C' : Type u₂
        inst✝ : CategoryTheory.Category.{v, u₂} C'
        D : CategoryTheory.GlueData' C
        i j k : D.J
        h✝ : Eq i j
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simp [*]
      /-
        🎉 no goals
      -/
      /-
        case pos
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v, u₁} C
        C' : Type u₂
        inst✝ : CategoryTheory.Category.{v, u₂} C'
        D : CategoryTheory.GlueData' C
        i j k : D.J
        h✝² : Not (Eq i j)
        h✝¹ : Eq i k
        h✝ : Eq j k
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (letFun ⋯ fun this => CategoryTheory. …
      -/
    · cases ‹i ≠ j› (‹i = k›.trans ‹j = k›.symm)
      /-
        🎉 no goals
      -/
      /-
        case neg
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v, u₁} C
        C' : Type u₂
        inst✝ : CategoryTheory.Category.{v, u₂} C'
        D : CategoryTheory.GlueData' C
        i j k : D.J
        h✝² : Not (Eq i j)
        h✝¹ : Eq i k
        h✝ : Not (Eq j k)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (letFun ⋯ fun this => CategoryTheory. …
      -/
    · simp [‹j ≠ k›.symm, *]
      /-
        🎉 no goals
      -/
      /-
        case pos
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v, u₁} C
        C' : Type u₂
        inst✝ : CategoryTheory.Category.{v, u₂} C'
        D : CategoryTheory.GlueData' C
        i j k : D.J
        h✝² : Not (Eq i j)
        h✝¹ : Not (Eq i k)
        h✝ : Eq j k
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (letFun ⋯ fun this => CategoryTheory. …
      -/
    · simp [*]
      /-
        🎉 no goals
      -/
      /-
        case neg
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v, u₁} C
        C' : Type u₂
        inst✝ : CategoryTheory.Category.{v, u₂} C'
        D : CategoryTheory.GlueData' C
        i j k : D.J
        h✝² : Not (Eq i j)
        h✝¹ : Not (Eq i k)
        h✝ : Not (Eq j k)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simp [*, reassoc_of% D.t_fac]
      /-
        🎉 no goals
      -/
  cocycle i j k := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v, u₁} C
      C' : Type u₂
      inst✝ : CategoryTheory.Category.{v, u₂} C'
      D : CategoryTheory.GlueData' C
      i j k : D.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.t'' i j k) (CategoryTheory.Categor …
    -/
    delta GlueData'.t''
    if hij : i = j then
      subst hij
      if hik : i = k then
        subst hik
        ext <;> simp
      else
        simp [hik, Ne.symm hik, fst_eq_snd_of_mono_eq]
    else if hik : i = k then
      subst hik
      ext <;> simp [hij, Ne.symm hij, fst_eq_snd_of_mono_eq, pullback.condition_assoc]
    else if hjk : j = k then
      subst hjk
      ext <;> simp [hij, Ne.symm hij, fst_eq_snd_of_mono_eq, pullback.condition_assoc]
    else
      ext <;> simp [hij, Ne.symm hij, hik, Ne.symm hik, hjk, Ne.symm hjk,
        pullback.map_comp_assoc]


