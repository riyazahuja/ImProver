/-- A path in a simplicial set `X` of length `n` is a directed path of `n` edges.-/
@[ext]
structure Path (n : ℕ) where
  /-- A path includes the data of `n+1` 0-simplices in `X`.-/
  vertex (i : Fin (n + 1)) : X _[0]
  /-- A path includes the data of `n` 1-simplices in `X`.-/
  arrow (i : Fin n) : X _[1]
  /-- The sources of the 1-simplices in a path are identified with appropriate 0-simplices.-/
  arrow_src (i : Fin n) : X.δ 1 (arrow i) = vertex i.castSucc
  /-- The targets of the 1-simplices in a path are identified with appropriate 0-simplices.-/
  arrow_tgt (i : Fin n) : X.δ 0 (arrow i) = vertex i.succ



variable {X} in
/-- For `j + l ≤ n`, a path of length `n` restricts to a path of length `l`, namely the subpath
spanned by the vertices `j ≤ i ≤ j + l` and edges `j ≤ i < j + l`. -/
def Path.interval {n : ℕ} (f : Path X n) (j l : ℕ) (hjl : j + l ≤ n) :
    Path X l where
                                  /-
                                    X : SSet
                                    n : Nat
                                    f : X.Path n
                                    j l : Nat
                                    hjl : LE.le (HAdd.hAdd j l) n
                                    i : Fin (HAdd.hAdd l 1)
                                    ⊢ LT.lt (HAdd.hAdd j ↑i) (HAdd.hAdd n 1)
                                  -/
  vertex i := f.vertex ⟨j + i, by omega⟩
                                  /-
                                    🎉 no goals
                                  -/
                                /-
                                  X : SSet
                                  n : Nat
                                  f : X.Path n
                                  j l : Nat
                                  hjl : LE.le (HAdd.hAdd j l) n
                                  i : Fin l
                                  ⊢ LT.lt (HAdd.hAdd j ↑i) n
                                -/
  arrow i := f.arrow ⟨j + i, by omega⟩
                                /-
                                  🎉 no goals
                                -/
                                        /-
                                          X : SSet
                                          n : Nat
                                          f : X.Path n
                                          j l : Nat
                                          hjl : LE.le (HAdd.hAdd j l) n
                                          i : Fin l
                                          ⊢ LT.lt (HAdd.hAdd j ↑i) n
                                        -/
  arrow_src i := f.arrow_src ⟨j + i, by omega⟩
                                        /-
                                          🎉 no goals
                                        -/
                                        /-
                                          X : SSet
                                          n : Nat
                                          f : X.Path n
                                          j l : Nat
                                          hjl : LE.le (HAdd.hAdd j l) n
                                          i : Fin l
                                          ⊢ LT.lt (HAdd.hAdd j ↑i) n
                                        -/
  arrow_tgt i := f.arrow_tgt ⟨j + i, by omega⟩
                                        /-
                                          🎉 no goals
                                        -/


/-- The spine of an `n`-simplex in `X` is the path of edges of length `n` formed by
traversing through its vertices in order.-/
@[simps]
def spine (n : ℕ) (Δ : X _[n]) : X.Path n where
  vertex i := X.map (SimplexCategory.const [0] [n] i).op Δ
  arrow i := X.map (SimplexCategory.mkOfSucc i).op Δ
  arrow_src i := by
    /-
      X : SSet
      n : Nat
      Δ : X.obj { unop := SimplexCategory.mk n }
      i : Fin n
      ⊢ Eq (CategoryTheory.SimplicialObject.δ X 1 ((fun i => X.map (SimplexCategory. …
    -/
    dsimp [SimplicialObject.δ]
    /-
      X : SSet
      n : Nat
      Δ : X.obj { unop := SimplexCategory.mk n }
      i : Fin n
      ⊢ Eq (X.map (SimplexCategory.δ 1).op (X.map (SimplexCategory.mkOfSucc i).op Δ) …
    -/
    simp only [← FunctorToTypes.map_comp_apply, ← op_comp]
    /-
      X : SSet
      n : Nat
      Δ : X.obj { unop := SimplexCategory.mk n }
      i : Fin n
      ⊢ Eq (X.map (CategoryTheory.CategoryStruct.comp (SimplexCategory.δ 1) (Simplex …
    -/
    rw [SimplexCategory.δ_one_mkOfSucc]
    /-
      X : SSet
      n : Nat
      Δ : X.obj { unop := SimplexCategory.mk n }
      i : Fin n
      ⊢ Eq (X.map ((SimplexCategory.mk 0).const (SimplexCategory.mk n) ↑↑i.castSucc) …
    -/
    simp only [len_mk, Fin.coe_castSucc, Fin.coe_eq_castSucc]
    /-
      🎉 no goals
    -/
  arrow_tgt i := by
    /-
      X : SSet
      n : Nat
      Δ : X.obj { unop := SimplexCategory.mk n }
      i : Fin n
      ⊢ Eq (CategoryTheory.SimplicialObject.δ X 0 ((fun i => X.map (SimplexCategory. …
    -/
    dsimp [SimplicialObject.δ]
    /-
      X : SSet
      n : Nat
      Δ : X.obj { unop := SimplexCategory.mk n }
      i : Fin n
      ⊢ Eq (X.map (SimplexCategory.δ 0).op (X.map (SimplexCategory.mkOfSucc i).op Δ) …
    -/
    simp only [← FunctorToTypes.map_comp_apply, ← op_comp]
    /-
      X : SSet
      n : Nat
      Δ : X.obj { unop := SimplexCategory.mk n }
      i : Fin n
      ⊢ Eq (X.map (CategoryTheory.CategoryStruct.comp (SimplexCategory.δ 0) (Simplex …
    -/
    rw [SimplexCategory.δ_zero_mkOfSucc]
    /-
      🎉 no goals
    -/


lemma spine_map_vertex {n : ℕ} (x : X _[n]) {m : ℕ} (φ : ([m] : SimplexCategory) ⟶ [n])
    (i : Fin (m + 1)) :
    (spine X m (X.map φ.op x)).vertex i = (spine X n x).vertex (φ.toOrderHom i) := by
  /-
    X : SSet
    n : Nat
    x : X.obj { unop := SimplexCategory.mk n }
    m : Nat
    φ : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk n)
    i : Fin (HAdd.hAdd m 1)
    ⊢ Eq ((X.spine m (X.map φ.op x)).vertex i) ((X.spine n x).vertex ((SimplexCate …
  -/
  dsimp [spine]
  /-
    X : SSet
    n : Nat
    x : X.obj { unop := SimplexCategory.mk n }
    m : Nat
    φ : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk n)
    i : Fin (HAdd.hAdd m 1)
    ⊢ Eq (X.map ((SimplexCategory.mk 0).const (SimplexCategory.mk m) i).op (X.map  …
  -/
  rw [← FunctorToTypes.map_comp_apply]
  /-
    X : SSet
    n : Nat
    x : X.obj { unop := SimplexCategory.mk n }
    m : Nat
    φ : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk n)
    i : Fin (HAdd.hAdd m 1)
    ⊢ Eq (X.map (CategoryTheory.CategoryStruct.comp φ.op ((SimplexCategory.mk 0).c …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma spine_map_subinterval {n : ℕ} (j l : ℕ) (hjl : j + l ≤ n) (Δ : X _[n]) :
                                          /-
                                            X : SSet
                                            n j l : Nat
                                            hjl : LE.le (HAdd.hAdd j l) n
                                            Δ : X.obj { unop := SimplexCategory.mk n }
                                            ⊢ LE.le (HAdd.hAdd j l) n
                                          -/
    X.spine l (X.map (subinterval j l (by omega)).op Δ) =
                                          /-
                                            🎉 no goals
                                          -/
                                     /-
                                       X : SSet
                                       n j l : Nat
                                       hjl : LE.le (HAdd.hAdd j l) n
                                       Δ : X.obj { unop := SimplexCategory.mk n }
                                       ⊢ LE.le (HAdd.hAdd j l) n
                                     -/
      (X.spine n Δ).interval j l (by omega) := by
                                     /-
                                       🎉 no goals
                                     -/
  /-
    X : SSet
    n j l : Nat
    hjl : LE.le (HAdd.hAdd j l) n
    Δ : X.obj { unop := SimplexCategory.mk n }
    ⊢ Eq (X.spine l (X.map (SimplexCategory.subinterval j l ⋯).op Δ)) ((X.spine n  …
  -/
  ext i
  · simp only [spine_vertex, Path.interval, ← FunctorToTypes.map_comp_apply, ← op_comp,
      const_subinterval_eq]
  · simp only [spine_arrow, Path.interval, ← FunctorToTypes.map_comp_apply, ← op_comp,
      mkOfSucc_subinterval_eq]


/-- Two paths of the same nonzero length are equal if all of their arrows are equal. -/
@[ext]
lemma Path.ext' {n : ℕ} {f g : Path X (n + 1)}
    (h : ∀ i : Fin (n + 1), f.arrow i = g.arrow i) :
    f = g := by
  /-
    X : SSet
    n : Nat
    f g : X.Path (HAdd.hAdd n 1)
    h : ∀ (i : Fin (HAdd.hAdd n 1)), Eq (f.arrow i) (g.arrow i)
    ⊢ Eq f g
  -/
  ext j
    /-
      case vertex.h
      X : SSet
      n : Nat
      f g : X.Path (HAdd.hAdd n 1)
      h : ∀ (i : Fin (HAdd.hAdd n 1)), Eq (f.arrow i) (g.arrow i)
      j : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
      ⊢ Eq (f.vertex j) (g.vertex j)
    -/
  · rcases Fin.eq_castSucc_or_eq_last j with ⟨k, hk⟩ | hl
      /-
        case vertex.h.inl.intro
        X : SSet
        n : Nat
        f g : X.Path (HAdd.hAdd n 1)
        h : ∀ (i : Fin (HAdd.hAdd n 1)), Eq (f.arrow i) (g.arrow i)
        j : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
        k : Fin (HAdd.hAdd n 1)
        hk : Eq j k.castSucc
        ⊢ Eq (f.vertex j) (g.vertex j)
      -/
    · rw [hk, ← f.arrow_src k, ← g.arrow_src k, h]
      /-
        🎉 no goals
      -/
      /-
        case vertex.h.inr
        X : SSet
        n : Nat
        f g : X.Path (HAdd.hAdd n 1)
        h : ∀ (i : Fin (HAdd.hAdd n 1)), Eq (f.arrow i) (g.arrow i)
        j : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
        hl : Eq j (Fin.last (HAdd.hAdd n 1))
        ⊢ Eq (f.vertex j) (g.vertex j)
      -/
    · simp only [hl, ← Fin.succ_last]
      /-
        case vertex.h.inr
        X : SSet
        n : Nat
        f g : X.Path (HAdd.hAdd n 1)
        h : ∀ (i : Fin (HAdd.hAdd n 1)), Eq (f.arrow i) (g.arrow i)
        j : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
        hl : Eq j (Fin.last (HAdd.hAdd n 1))
        ⊢ Eq (f.vertex (Fin.last n).succ) (g.vertex (Fin.last n).succ)
      -/
      rw [← f.arrow_tgt (Fin.last n), ← g.arrow_tgt (Fin.last n), h]
      /-
        🎉 no goals
      -/
    /-
      case arrow.h
      X : SSet
      n : Nat
      f g : X.Path (HAdd.hAdd n 1)
      h : ∀ (i : Fin (HAdd.hAdd n 1)), Eq (f.arrow i) (g.arrow i)
      j : Fin (HAdd.hAdd n 1)
      ⊢ Eq (f.arrow j) (g.arrow j)
    -/
  · exact h j
    /-
      🎉 no goals
    -/


/-- Maps of simplicial sets induce maps of paths in a simplicial set.-/
@[simps]
def Path.map {X Y : SSet.{u}} {n : ℕ} (f : X.Path n) (σ : X ⟶ Y) : Y.Path n where
  vertex i := σ.app (Opposite.op [0]) (f.vertex i)
  arrow i := σ.app (Opposite.op [1]) (f.arrow i)
  arrow_src i := by
    /-
      X✝ X Y : SSet
      n : Nat
      f : X.Path n
      σ : Quiver.Hom X Y
      i : Fin n
      ⊢ Eq (CategoryTheory.SimplicialObject.δ Y 1 ((fun i => σ.app { unop := Simplex …
    -/
    simp only [← f.arrow_src i]
    /-
      X✝ X Y : SSet
      n : Nat
      f : X.Path n
      σ : Quiver.Hom X Y
      i : Fin n
      ⊢ Eq (CategoryTheory.SimplicialObject.δ Y 1 (σ.app { unop := SimplexCategory.m …
    -/
    exact congr (σ.naturality (δ 1).op) rfl |>.symm
    /-
      🎉 no goals
    -/
  arrow_tgt i := by
    /-
      X✝ X Y : SSet
      n : Nat
      f : X.Path n
      σ : Quiver.Hom X Y
      i : Fin n
      ⊢ Eq (CategoryTheory.SimplicialObject.δ Y 0 ((fun i => σ.app { unop := Simplex …
    -/
    simp only [← f.arrow_tgt i]
    /-
      X✝ X Y : SSet
      n : Nat
      f : X.Path n
      σ : Quiver.Hom X Y
      i : Fin n
      ⊢ Eq (CategoryTheory.SimplicialObject.δ Y 0 (σ.app { unop := SimplexCategory.m …
    -/
    exact congr (σ.naturality (δ 0).op) rfl |>.symm
    /-
      🎉 no goals
    -/


/-- `Path.map` respects subintervals of paths.-/
lemma map_interval {X Y : SSet.{u}} {n : ℕ} (f : X.Path n) (σ : X ⟶ Y)
    (j l : ℕ) (hjl : j + l ≤ n) :
    (f.map σ).interval j l hjl = (f.interval j l hjl).map σ := rfl


/-- The spine of the unique non-degenerate `n`-simplex in `Δ[n]`.-/
def standardSimplex.spineId (n : ℕ) : Path Δ[n] n :=
  spine Δ[n] n (standardSimplex.id n)


/-- Any inner horn contains the spine of the unique non-degenerate `n`-simplex
in `Δ[n]`.-/
@[simps]
def horn.spineId {n : ℕ} (i : Fin (n + 3))
    (h₀ : 0 < i) (hₙ : i < Fin.last (n + 2)) :
    Path Λ[n + 2, i] (n + 2) where
  vertex j := ⟨standardSimplex.spineId _ |>.vertex j, (horn.const n i j _).property⟩
  arrow j := ⟨standardSimplex.spineId _ |>.arrow j, by
    /-
      X : SSet
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
      j : Fin (HAdd.hAdd n 2)
      ⊢ Ne (Union.union (Set.range ⇑(SSet.asOrderHom ((SSet.standardSimplex.spineId  …
    -/
    let edge := horn.primitiveEdge h₀ hₙ j
    have ha : (standardSimplex.spineId _).arrow j = edge.val := by
      dsimp only [edge, standardSimplex.spineId, standardSimplex.id, spine_arrow,
        mkOfSucc, horn.primitiveEdge, horn.edge, standardSimplex.edge,
        standardSimplex.map_apply]
      aesop
    /-
      X : SSet
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
      j : Fin (HAdd.hAdd n 2)
      edge : (SSet.horn (HAdd.hAdd n 2) i).obj { unop := SimplexCategory.mk 1 } := S …
      ha : Eq ((SSet.standardSimplex.spineId (HAdd.hAdd n 2)).arrow j) ↑edge
      ⊢ Ne (Union.union (Set.range ⇑(SSet.asOrderHom ((SSet.standardSimplex.spineId  …
    -/
    rw [ha]
    /-
      X : SSet
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
      j : Fin (HAdd.hAdd n 2)
      edge : (SSet.horn (HAdd.hAdd n 2) i).obj { unop := SimplexCategory.mk 1 } := S …
      ha : Eq ((SSet.standardSimplex.spineId (HAdd.hAdd n 2)).arrow j) ↑edge
      ⊢ Ne (Union.union (Set.range ⇑(SSet.asOrderHom ↑edge)) (Singleton.singleton i) …
    -/
    exact edge.property⟩
    /-
      🎉 no goals
    -/
  arrow_src := by
    /-
      X : SSet
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
      ⊢ ∀ (i_1 : Fin (HAdd.hAdd n 2)), Eq (CategoryTheory.SimplicialObject.δ (SSet.h …
    -/
    simp only [horn, SimplicialObject.δ, Subtype.mk.injEq]
    /-
      X : SSet
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
      ⊢ ∀ (i : Fin (HAdd.hAdd n 2)), Eq ((SSet.standardSimplex.obj (SimplexCategory. …
    -/
    exact standardSimplex.spineId _ |>.arrow_src
    /-
      🎉 no goals
    -/
  arrow_tgt := by
    /-
      X : SSet
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
      ⊢ ∀ (i_1 : Fin (HAdd.hAdd n 2)), Eq (CategoryTheory.SimplicialObject.δ (SSet.h …
    -/
    simp only [horn, SimplicialObject.δ, Subtype.mk.injEq]
    /-
      X : SSet
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
      ⊢ ∀ (i : Fin (HAdd.hAdd n 2)), Eq ((SSet.standardSimplex.obj (SimplexCategory. …
    -/
    exact standardSimplex.spineId _ |>.arrow_tgt
    /-
      🎉 no goals
    -/


@[simp]
lemma horn.spineId_map_hornInclusion {n : ℕ} (i : Fin (n + 3))
    (h₀ : 0 < i) (hₙ : i < Fin.last (n + 2)) :
    Path.map (horn.spineId i h₀ hₙ) (hornInclusion (n + 2) i) =
      standardSimplex.spineId (n + 2) := rfl


