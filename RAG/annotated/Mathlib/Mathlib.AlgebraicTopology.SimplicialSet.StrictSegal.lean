/-- A simplicial set `X` satisfies the strict Segal condition if its simplices are uniquely
determined by their spine. -/
class StrictSegal where
  /-- The inverse to `X.spine n`.-/
  spineToSimplex {n : ℕ} : Path X n → X _[n]
  /-- `spineToSimplex` is a right inverse to `X.spine n`.-/
  spine_spineToSimplex {n : ℕ} (f : Path X n) : X.spine n (spineToSimplex f) = f
  /-- `spineToSimplex` is a left inverse to `X.spine n`.-/
  spineToSimplex_spine {n : ℕ} (Δ : X _[n]) : spineToSimplex (X.spine n Δ) = Δ


/-- The fields of `StrictSegal` define an equivalence between `X _[n]` and `Path X n`.-/
def spineEquiv (n : ℕ) : X _[n] ≃ Path X n where
  toFun := spine X n
  invFun := spineToSimplex
  left_inv := spineToSimplex_spine
  right_inv := spine_spineToSimplex


theorem spineInjective {n : ℕ} : Function.Injective (spineEquiv (X := X) n) := Equiv.injective _


@[simp]
theorem spineToSimplex_vertex (i : Fin (n + 1)) (f : Path X n) :
    X.map (const [0] [n] i).op (spineToSimplex f) = f.vertex i := by
  /-
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    f : X.Path n
    ⊢ Eq (X.map ((SimplexCategory.mk 0).const (SimplexCategory.mk n) i).op (SSet.S …
  -/
  rw [← spine_vertex, spine_spineToSimplex]
  /-
    🎉 no goals
  -/


@[simp]
theorem spineToSimplex_arrow (i : Fin n) (f : Path X n) :
    X.map (mkOfSucc i).op (spineToSimplex f) = f.arrow i := by
  /-
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    i : Fin n
    f : X.Path n
    ⊢ Eq (X.map (SimplexCategory.mkOfSucc i).op (SSet.StrictSegal.spineToSimplex f …
  -/
  rw [← spine_arrow, spine_spineToSimplex]
  /-
    🎉 no goals
  -/


/-- In the presence of the strict Segal condition, a path of length `n` can be "composed" by taking
the diagonal edge of the resulting `n`-simplex. -/
def spineToDiagonal (f : Path X n) : X _[1] := diagonal X (spineToSimplex f)


@[simp]
theorem spineToSimplex_interval (f : Path X n) (j l : ℕ) (hjl : j + l ≤  n)  :
    X.map (subinterval j l hjl).op (spineToSimplex f) =
      spineToSimplex (Path.interval f j l hjl) := by
  /-
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    f : X.Path n
    j l : Nat
    hjl : LE.le (HAdd.hAdd j l) n
    ⊢ Eq (X.map (SimplexCategory.subinterval j l hjl).op (SSet.StrictSegal.spineTo …
  -/
  apply spineInjective
  /-
    case a
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    f : X.Path n
    j l : Nat
    hjl : LE.le (HAdd.hAdd j l) n
    ⊢ Eq ((SSet.StrictSegal.spineEquiv l) (X.map (SimplexCategory.subinterval j l  …
  -/
  unfold spineEquiv
  /-
    case a
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    f : X.Path n
    j l : Nat
    hjl : LE.le (HAdd.hAdd j l) n
    ⊢ Eq ({ toFun := X.spine l, invFun := SSet.StrictSegal.spineToSimplex, left_in …
  -/
  dsimp
  /-
    case a
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    f : X.Path n
    j l : Nat
    hjl : LE.le (HAdd.hAdd j l) n
    ⊢ Eq (X.spine l (X.map (SimplexCategory.subinterval j l hjl).op (SSet.StrictSe …
  -/
  rw [spine_spineToSimplex]
  /-
    case a
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    f : X.Path n
    j l : Nat
    hjl : LE.le (HAdd.hAdd j l) n
    ⊢ Eq (X.spine l (X.map (SimplexCategory.subinterval j l hjl).op (SSet.StrictSe …
  -/
  convert spine_map_subinterval X j l hjl (spineToSimplex f)
  /-
    case h.e'_3.h.e'_3
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    f : X.Path n
    j l : Nat
    hjl : LE.le (HAdd.hAdd j l) n
    ⊢ Eq f (X.spine n (SSet.StrictSegal.spineToSimplex f))
  -/
  exact Eq.symm (spine_spineToSimplex f)
  /-
    🎉 no goals
  -/


theorem spineToSimplex_edge (f : Path X n) (j l : ℕ) (hjl : j + l ≤ n) :
    X.map (intervalEdge j l hjl).op (spineToSimplex f) =
      spineToDiagonal (Path.interval f j l hjl) := by
  /-
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    f : X.Path n
    j l : Nat
    hjl : LE.le (HAdd.hAdd j l) n
    ⊢ Eq (X.map (SimplexCategory.intervalEdge j l hjl).op (SSet.StrictSegal.spineT …
  -/
  unfold spineToDiagonal
  /-
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    f : X.Path n
    j l : Nat
    hjl : LE.le (HAdd.hAdd j l) n
    ⊢ Eq (X.map (SimplexCategory.intervalEdge j l hjl).op (SSet.StrictSegal.spineT …
  -/
  rw [← congrArg (diagonal X) (spineToSimplex_interval f j l hjl)]
  /-
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    f : X.Path n
    j l : Nat
    hjl : LE.le (HAdd.hAdd j l) n
    ⊢ Eq (X.map (SimplexCategory.intervalEdge j l hjl).op (SSet.StrictSegal.spineT …
  -/
  unfold diagonal
  /-
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    f : X.Path n
    j l : Nat
    hjl : LE.le (HAdd.hAdd j l) n
    ⊢ Eq (X.map (SimplexCategory.intervalEdge j l hjl).op (SSet.StrictSegal.spineT …
  -/
  simp only [← FunctorToTypes.map_comp_apply, ← op_comp, diag_subinterval_eq]
  /-
    🎉 no goals
  -/


/-- For any `σ : X ⟶ Y` between `StrictSegal` simplicial sets, `spineToSimplex`
commutes with `Path.map`. -/
lemma spineToSimplex_map {X Y : SSet.{u}} [StrictSegal X] [StrictSegal Y]
    {n : ℕ} (f : Path X (n + 1)) (σ : X ⟶ Y) :
    spineToSimplex (f.map σ) = σ.app _ (spineToSimplex f) := by
  /-
    X Y : SSet
    inst✝¹ : X.StrictSegal
    inst✝ : Y.StrictSegal
    n : Nat
    f : X.Path (HAdd.hAdd n 1)
    σ : Quiver.Hom X Y
    ⊢ Eq (SSet.StrictSegal.spineToSimplex (f.map σ)) (σ.app { unop := SimplexCateg …
  -/
  apply spineInjective
  /-
    case a
    X Y : SSet
    inst✝¹ : X.StrictSegal
    inst✝ : Y.StrictSegal
    n : Nat
    f : X.Path (HAdd.hAdd n 1)
    σ : Quiver.Hom X Y
    ⊢ Eq ((SSet.StrictSegal.spineEquiv (HAdd.hAdd n 1)) (SSet.StrictSegal.spineToS …
  -/
  ext k
  /-
    case a.h
    X Y : SSet
    inst✝¹ : X.StrictSegal
    inst✝ : Y.StrictSegal
    n : Nat
    f : X.Path (HAdd.hAdd n 1)
    σ : Quiver.Hom X Y
    k : Fin (HAdd.hAdd n 1)
    ⊢ Eq (((SSet.StrictSegal.spineEquiv (HAdd.hAdd n 1)) (SSet.StrictSegal.spineTo …
  -/
  dsimp only [spineEquiv, Equiv.coe_fn_mk, Path.map, spine_arrow]
  /-
    case a.h
    X Y : SSet
    inst✝¹ : X.StrictSegal
    inst✝ : Y.StrictSegal
    n : Nat
    f : X.Path (HAdd.hAdd n 1)
    σ : Quiver.Hom X Y
    k : Fin (HAdd.hAdd n 1)
    ⊢ Eq (Y.map (SimplexCategory.mkOfSucc k).op (SSet.StrictSegal.spineToSimplex { …
  -/
  rw [← types_comp_apply (σ.app _) (Y.map _), ← σ.naturality]
  /-
    case a.h
    X Y : SSet
    inst✝¹ : X.StrictSegal
    inst✝ : Y.StrictSegal
    n : Nat
    f : X.Path (HAdd.hAdd n 1)
    σ : Quiver.Hom X Y
    k : Fin (HAdd.hAdd n 1)
    ⊢ Eq (Y.map (SimplexCategory.mkOfSucc k).op (SSet.StrictSegal.spineToSimplex { …
  -/
  simp only [types_comp_apply, spineToSimplex_arrow]
  /-
    🎉 no goals
  -/


/-- If we take the path along the spine of the `j`th face of a `spineToSimplex`,
the common vertices will agree with those of the original path `f`. In particular,
a vertex `i` with `i < j` can be identified with the same vertex in `f`. -/
lemma spine_δ_vertex_lt (f : Path X (n + 1)) {i : Fin (n + 1)} {j : Fin (n + 2)}
    (h : i.castSucc < j) :
    (X.spine n (X.δ j (spineToSimplex f))).vertex i = f.vertex i.castSucc := by
  /-
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    f : X.Path (HAdd.hAdd n 1)
    i : Fin (HAdd.hAdd n 1)
    j : Fin (HAdd.hAdd n 2)
    h : LT.lt i.castSucc j
    ⊢ Eq ((X.spine n (CategoryTheory.SimplicialObject.δ X j (SSet.StrictSegal.spin …
  -/
  simp only [SimplicialObject.δ, spine_vertex]
  /-
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    f : X.Path (HAdd.hAdd n 1)
    i : Fin (HAdd.hAdd n 1)
    j : Fin (HAdd.hAdd n 2)
    h : LT.lt i.castSucc j
    ⊢ Eq (X.map ((SimplexCategory.mk 0).const (SimplexCategory.mk n) i).op (X.map  …
  -/
  rw [← FunctorToTypes.map_comp_apply, ← op_comp, const_comp, spineToSimplex_vertex]
  simp only [SimplexCategory.δ, Hom.toOrderHom, len_mk, mkHom, Hom.mk,
    OrderEmbedding.toOrderHom_coe, Fin.succAboveOrderEmb_apply]
  /-
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    f : X.Path (HAdd.hAdd n 1)
    i : Fin (HAdd.hAdd n 1)
    j : Fin (HAdd.hAdd n 2)
    h : LT.lt i.castSucc j
    ⊢ Eq (f.vertex (j.succAbove i)) (f.vertex i.castSucc)
  -/
  rw [Fin.succAbove_of_castSucc_lt j i h]
  /-
    🎉 no goals
  -/


/-- If we take the path along the spine of the `j`th face of a `spineToSimplex`,
a vertex `i` with `i ≥ j` can be identified with vertex `i + 1` in the original
path. -/
lemma spine_δ_vertex_ge (f : Path X (n + 1)) {i : Fin (n + 1)} {j : Fin (n + 2)}
    (h : j ≤ i.castSucc) :
    (X.spine n (X.δ j (spineToSimplex f))).vertex i = f.vertex i.succ := by
  /-
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    f : X.Path (HAdd.hAdd n 1)
    i : Fin (HAdd.hAdd n 1)
    j : Fin (HAdd.hAdd n 2)
    h : LE.le j i.castSucc
    ⊢ Eq ((X.spine n (CategoryTheory.SimplicialObject.δ X j (SSet.StrictSegal.spin …
  -/
  simp only [SimplicialObject.δ, spine_vertex]
  /-
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    f : X.Path (HAdd.hAdd n 1)
    i : Fin (HAdd.hAdd n 1)
    j : Fin (HAdd.hAdd n 2)
    h : LE.le j i.castSucc
    ⊢ Eq (X.map ((SimplexCategory.mk 0).const (SimplexCategory.mk n) i).op (X.map  …
  -/
  rw [← FunctorToTypes.map_comp_apply, ← op_comp, const_comp, spineToSimplex_vertex]
  simp only [SimplexCategory.δ, Hom.toOrderHom, len_mk, mkHom, Hom.mk,
    OrderEmbedding.toOrderHom_coe, Fin.succAboveOrderEmb_apply]
  /-
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    f : X.Path (HAdd.hAdd n 1)
    i : Fin (HAdd.hAdd n 1)
    j : Fin (HAdd.hAdd n 2)
    h : LE.le j i.castSucc
    ⊢ Eq (f.vertex (j.succAbove i)) (f.vertex i.succ)
  -/
  rw [Fin.succAbove_of_le_castSucc j i h]
  /-
    🎉 no goals
  -/


/-- If we take the path along the spine of the `j`th face of a `spineToSimplex`,
the common arrows will agree with those of the original path `f`. In particular,
an arrow `i` with `i + 1 < j` can be identified with the same arrow in `f`. -/
lemma spine_δ_arrow_lt (f : Path X (n + 1)) {i : Fin n} {j : Fin (n + 2)}
    (h : i.succ.castSucc < j) :
    (X.spine n (X.δ j (spineToSimplex f))).arrow i = f.arrow i.castSucc := by
  /-
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    f : X.Path (HAdd.hAdd n 1)
    i : Fin n
    j : Fin (HAdd.hAdd n 2)
    h : LT.lt i.succ.castSucc j
    ⊢ Eq ((X.spine n (CategoryTheory.SimplicialObject.δ X j (SSet.StrictSegal.spin …
  -/
  simp only [SimplicialObject.δ, spine_arrow]
  /-
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    f : X.Path (HAdd.hAdd n 1)
    i : Fin n
    j : Fin (HAdd.hAdd n 2)
    h : LT.lt i.succ.castSucc j
    ⊢ Eq (X.map (SimplexCategory.mkOfSucc i).op (X.map (SimplexCategory.δ j).op (S …
  -/
  rw [← FunctorToTypes.map_comp_apply, ← op_comp]
  /-
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    f : X.Path (HAdd.hAdd n 1)
    i : Fin n
    j : Fin (HAdd.hAdd n 2)
    h : LT.lt i.succ.castSucc j
    ⊢ Eq (X.map (CategoryTheory.CategoryStruct.comp (SimplexCategory.mkOfSucc i) ( …
  -/
  rw [mkOfSucc_δ_lt h, spineToSimplex_arrow]
  /-
    🎉 no goals
  -/


/-- If we take the path along the spine of the `j`th face of a `spineToSimplex`,
an arrow `i` with `i + 1 > j` can be identified with arrow `i + 1` in the
original path. -/
lemma spine_δ_arrow_gt (f : Path X (n + 1)) {i : Fin n} {j : Fin (n + 2)}
    (h : j < i.succ.castSucc) :
    (X.spine n (X.δ j (spineToSimplex f))).arrow i = f.arrow i.succ := by
  /-
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    f : X.Path (HAdd.hAdd n 1)
    i : Fin n
    j : Fin (HAdd.hAdd n 2)
    h : LT.lt j i.succ.castSucc
    ⊢ Eq ((X.spine n (CategoryTheory.SimplicialObject.δ X j (SSet.StrictSegal.spin …
  -/
  simp only [SimplicialObject.δ, spine_arrow]
  /-
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    f : X.Path (HAdd.hAdd n 1)
    i : Fin n
    j : Fin (HAdd.hAdd n 2)
    h : LT.lt j i.succ.castSucc
    ⊢ Eq (X.map (SimplexCategory.mkOfSucc i).op (X.map (SimplexCategory.δ j).op (S …
  -/
  rw [← FunctorToTypes.map_comp_apply, ← op_comp]
  /-
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    f : X.Path (HAdd.hAdd n 1)
    i : Fin n
    j : Fin (HAdd.hAdd n 2)
    h : LT.lt j i.succ.castSucc
    ⊢ Eq (X.map (CategoryTheory.CategoryStruct.comp (SimplexCategory.mkOfSucc i) ( …
  -/
  rw [mkOfSucc_δ_gt h, spineToSimplex_arrow]
  /-
    🎉 no goals
  -/


/-- If we take the path along the spine of a face of a `spineToSimplex`, the
arrows not contained in the original path can be recovered as the diagonal edge
of the `spineToSimplex` that "composes" arrows `i` and `i + 1`. -/
lemma spine_δ_arrow_eq (f : Path X (n + 1)) {i : Fin n} {j : Fin (n + 2)}
    (h : j = i.succ.castSucc) :
    (X.spine n (X.δ j (spineToSimplex f))).arrow i =
                                               /-
                                                 X✝ X : SSet
                                                 inst✝ : X.StrictSegal
                                                 n : Nat
                                                 f : X.Path (HAdd.hAdd n 1)
                                                 i : Fin n
                                                 j : Fin (HAdd.hAdd n 2)
                                                 h : Eq j i.succ.castSucc
                                                 ⊢ LE.le (HAdd.hAdd (↑i) 2) (HAdd.hAdd n 1)
                                               -/
      spineToDiagonal (Path.interval f i 2 (by omega)) := by
                                               /-
                                                 🎉 no goals
                                               -/
  /-
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    f : X.Path (HAdd.hAdd n 1)
    i : Fin n
    j : Fin (HAdd.hAdd n 2)
    h : Eq j i.succ.castSucc
    ⊢ Eq ((X.spine n (CategoryTheory.SimplicialObject.δ X j (SSet.StrictSegal.spin …
  -/
  simp only [SimplicialObject.δ, spine_arrow]
  /-
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    f : X.Path (HAdd.hAdd n 1)
    i : Fin n
    j : Fin (HAdd.hAdd n 2)
    h : Eq j i.succ.castSucc
    ⊢ Eq (X.map (SimplexCategory.mkOfSucc i).op (X.map (SimplexCategory.δ j).op (S …
  -/
  rw [← FunctorToTypes.map_comp_apply, ← op_comp]
  /-
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    f : X.Path (HAdd.hAdd n 1)
    i : Fin n
    j : Fin (HAdd.hAdd n 2)
    h : Eq j i.succ.castSucc
    ⊢ Eq (X.map (CategoryTheory.CategoryStruct.comp (SimplexCategory.mkOfSucc i) ( …
  -/
  rw [mkOfSucc_δ_eq h, spineToSimplex_edge]
  /-
    🎉 no goals
  -/


/-- Simplices in the nerve of categories are uniquely determined by their spine. Indeed, this
property describes the essential image of the nerve functor.-/
noncomputable instance strictSegal (C : Type u) [Category.{v} C] : StrictSegal (nerve C) where
  spineToSimplex {n} F :=
    ComposableArrows.mkOfObjOfMapSucc (fun i ↦ (F.vertex i).obj 0)
      (fun i ↦ eqToHom (Functor.congr_obj (F.arrow_src i).symm 0) ≫
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          n : Nat
          F : (CategoryTheory.nerve C).Path n
          i : Fin (Opposite.unop { unop := SimplexCategory.mk n }).len
          ⊢ LE.le 0 1
        -/
        /-
          🎉 no goals
        -/
        (F.arrow i).map' 0 1 ≫ eqToHom (Functor.congr_obj (F.arrow_tgt i) 0))
        /-
          🎉 no goals
        -/
  spine_spineToSimplex {n} F := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      n : Nat
      F : (CategoryTheory.nerve C).Path n
      ⊢ Eq ((CategoryTheory.nerve C).spine n ((fun {n} F => CategoryTheory.Composabl …
    -/
    ext i
      /-
        case vertex.h
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        n : Nat
        F : (CategoryTheory.nerve C).Path n
        i : Fin (HAdd.hAdd n 1)
        ⊢ Eq (((CategoryTheory.nerve C).spine n ((fun {n} F => CategoryTheory.Composab …
      -/
    · exact ComposableArrows.ext₀ rfl
      /-
        🎉 no goals
      -/
      /-
        case arrow.h
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        n : Nat
        F : (CategoryTheory.nerve C).Path n
        i : Fin n
        ⊢ Eq (((CategoryTheory.nerve C).spine n ((fun {n} F => CategoryTheory.Composab …
      -/
    · refine ComposableArrows.ext₁ ?_ ?_ ?_
        /-
          case arrow.h.refine_1
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          n : Nat
          F : (CategoryTheory.nerve C).Path n
          i : Fin n
          ⊢ Eq (CategoryTheory.ComposableArrows.left (((CategoryTheory.nerve C).spine n  …
        -/
      · exact Functor.congr_obj (F.arrow_src i).symm 0
        /-
          🎉 no goals
        -/
        /-
          case arrow.h.refine_2
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          n : Nat
          F : (CategoryTheory.nerve C).Path n
          i : Fin n
          ⊢ Eq (CategoryTheory.ComposableArrows.right (((CategoryTheory.nerve C).spine n …
        -/
      · exact Functor.congr_obj (F.arrow_tgt i).symm 0
        /-
          🎉 no goals
        -/
        /-
          case arrow.h.refine_3
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          n : Nat
          F : (CategoryTheory.nerve C).Path n
          i : Fin n
          ⊢ Eq (CategoryTheory.ComposableArrows.hom (((CategoryTheory.nerve C).spine n ( …
        -/
      · dsimp
        /-
          case arrow.h.refine_3
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          n : Nat
          F : (CategoryTheory.nerve C).Path n
          i : Fin n
          ⊢ Eq ((CategoryTheory.ComposableArrows.mkOfObjOfMapSucc (fun i => (F.vertex i) …
        -/
        apply ComposableArrows.mkOfObjOfMapSucc_map_succ
        /-
          🎉 no goals
        -/
  spineToSimplex_spine {n} F := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      n : Nat
      F : (CategoryTheory.nerve C).obj { unop := SimplexCategory.mk n }
      ⊢ Eq ((fun {n} F => CategoryTheory.ComposableArrows.mkOfObjOfMapSucc (fun i => …
    -/
    fapply ComposableArrows.ext
      /-
        case h
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        n : Nat
        F : (CategoryTheory.nerve C).obj { unop := SimplexCategory.mk n }
        ⊢ ∀ (i : Fin (HAdd.hAdd (Opposite.unop { unop := SimplexCategory.mk n }).len 1 …
      -/
    · intro i
      /-
        case h
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        n : Nat
        F : (CategoryTheory.nerve C).obj { unop := SimplexCategory.mk n }
        i : Fin (HAdd.hAdd (Opposite.unop { unop := SimplexCategory.mk n }).len 1)
        ⊢ Eq (((fun {n} F => CategoryTheory.ComposableArrows.mkOfObjOfMapSucc (fun i = …
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case w
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        n : Nat
        F : (CategoryTheory.nerve C).obj { unop := SimplexCategory.mk n }
        ⊢ ∀ (i : Nat) (hi : LT.lt i (Opposite.unop { unop := SimplexCategory.mk n }).l …
      -/
    · intro i hi
      /-
        case w
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        n : Nat
        F : (CategoryTheory.nerve C).obj { unop := SimplexCategory.mk n }
        i : Nat
        hi : LT.lt i (Opposite.unop { unop := SimplexCategory.mk n }).len
        ⊢ Eq (((fun {n} F => CategoryTheory.ComposableArrows.mkOfObjOfMapSucc (fun i = …
      -/
      apply ComposableArrows.mkOfObjOfMapSucc_map_succ
      /-
        🎉 no goals
      -/


