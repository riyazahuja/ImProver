/-- The category of simplicial sets.
This is the category of contravariant functors from
`SimplexCategory` to `Type u`. -/
def SSet : Type (u + 1) :=
  SimplicialObject (Type u)


instance largeCategory : LargeCategory SSet := by
  /-
    ⊢ CategoryTheory.LargeCategory SSet
  -/
  dsimp only [SSet]
  /-
    ⊢ CategoryTheory.LargeCategory (CategoryTheory.SimplicialObject (Type ?u.18))
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance hasLimits : HasLimits SSet := by
  /-
    ⊢ CategoryTheory.Limits.HasLimits SSet
  -/
  dsimp only [SSet]
  /-
    ⊢ CategoryTheory.Limits.HasLimits (CategoryTheory.SimplicialObject (Type u_1))
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance hasColimits : HasColimits SSet := by
  /-
    ⊢ CategoryTheory.Limits.HasColimits SSet
  -/
  dsimp only [SSet]
  /-
    ⊢ CategoryTheory.Limits.HasColimits (CategoryTheory.SimplicialObject (Type u_1))
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[ext]
lemma hom_ext {X Y : SSet} {f g : X ⟶ Y} (w : ∀ n, f.app n = g.app n) : f = g :=
  SimplicialObject.hom_ext _ _ w


@[simp]
lemma comp_app {X Y Z : SSet} (f : X ⟶ Y) (g : Y ⟶ Z) (n : SimplexCategoryᵒᵖ) :
    (f ≫ g).app n = f.app n ≫ g.app n := NatTrans.comp_app _ _ _


/-- The ulift functor `SSet.{u} ⥤ SSet.{max u v}` on simplicial sets. -/
def uliftFunctor : SSet.{u} ⥤ SSet.{max u v} :=
  (SimplicialObject.whiskering _ _).obj CategoryTheory.uliftFunctor.{v, u}


/-- The `n`-th standard simplex `Δ[n]` associated with a nonempty finite linear order `n`
is the Yoneda embedding of `n`. -/
def standardSimplex : SimplexCategory ⥤ SSet.{u} :=
  yoneda ⋙ uliftFunctor


@[inherit_doc SSet.standardSimplex]
scoped[Simplicial] notation3 "Δ[" n "]" => SSet.standardSimplex.obj (SimplexCategory.mk n)


instance : Inhabited SSet :=
  ⟨Δ[0]⟩


@[simp]
lemma map_id (n : SimplexCategory) :
    (SSet.standardSimplex.map (SimplexCategory.Hom.mk OrderHom.id : n ⟶ n)) = 𝟙 _ :=
  CategoryTheory.Functor.map_id _ _


/-- Simplices of the standard simplex identify to morphisms in `SimplexCategory`. -/
def objEquiv (n : SimplexCategory) (m : SimplexCategoryᵒᵖ) :
    (standardSimplex.{u}.obj n).obj m ≃ (m.unop ⟶ n) :=
  Equiv.ulift.{u, 0}


/-- Constructor for simplices of the standard simplex which takes a `OrderHom` as an input. -/
abbrev objMk {n : SimplexCategory} {m : SimplexCategoryᵒᵖ}
    (f : Fin (len m.unop + 1) →o Fin (n.len + 1)) :
    (standardSimplex.{u}.obj n).obj m :=
  (objEquiv _ _).symm (Hom.mk f)


lemma map_apply {m₁ m₂ : SimplexCategoryᵒᵖ} (f : m₁ ⟶ m₂) {n : SimplexCategory}
    (x : (standardSimplex.{u}.obj n).obj m₁) :
    (standardSimplex.{u}.obj n).map f x = (objEquiv _ _).symm (f.unop ≫ (objEquiv _ _) x) := by
  /-
    m₁ m₂ : Opposite SimplexCategory
    f : Quiver.Hom m₁ m₂
    n : SimplexCategory
    x : (SSet.standardSimplex.obj n).obj m₁
    ⊢ Eq ((SSet.standardSimplex.obj n).map f x) ((SSet.standardSimplex.objEquiv n  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The canonical bijection `(standardSimplex.obj n ⟶ X) ≃ X.obj (op n)`. -/
def _root_.SSet.yonedaEquiv (X : SSet.{u}) (n : SimplexCategory) :
    (standardSimplex.obj n ⟶ X) ≃ X.obj (op n) :=
  yonedaCompUliftFunctorEquiv X n


/-- The unique non-degenerate `n`-simplex in `Δ[n]`. -/
def id (n : ℕ) : Δ[n] _[n] := yonedaEquiv Δ[n] [n] (𝟙 Δ[n])


lemma id_eq_objEquiv_symm (n : ℕ) : id n = (objEquiv _ _).symm (𝟙 _) := rfl


lemma objEquiv_id (n : ℕ) : objEquiv _ _ (id n) = 𝟙 _ := rfl


/-- The (degenerate) `m`-simplex in the standard simplex concentrated in vertex `k`. -/
def const (n : ℕ) (k : Fin (n+1)) (m : SimplexCategoryᵒᵖ) : Δ[n].obj m :=
  objMk (OrderHom.const _ k )


@[simp]
lemma const_down_toOrderHom (n : ℕ) (k : Fin (n+1)) (m : SimplexCategoryᵒᵖ) :
    (const n k m).down.toOrderHom = OrderHom.const _ k :=
  rfl


/-- The edge of the standard simplex with endpoints `a` and `b`. -/
def edge (n : ℕ) (a b : Fin (n+1)) (hab : a ≤ b) : Δ[n] _[1] := by
  /-
    n : Nat
    a b : Fin (HAdd.hAdd n 1)
    hab : LE.le a b
    ⊢ (SSet.standardSimplex.obj (SimplexCategory.mk n)).obj { unop := SimplexCateg …
  -/
  refine objMk ⟨![a, b], ?_⟩
  /-
    n : Nat
    a b : Fin (HAdd.hAdd n 1)
    hab : LE.le a b
    ⊢ Monotone (Matrix.vecCons a (Matrix.vecCons b Matrix.vecEmpty))
  -/
  rw [Fin.monotone_iff_le_succ]
  /-
    n : Nat
    a b : Fin (HAdd.hAdd n 1)
    hab : LE.le a b
    ⊢ ∀ (i : Fin (Opposite.unop { unop := SimplexCategory.mk 1 }).len), LE.le (Mat …
  -/
  simp only [unop_op, len_mk, Fin.forall_fin_one]
  /-
    n : Nat
    a b : Fin (HAdd.hAdd n 1)
    hab : LE.le a b
    ⊢ LE.le (Matrix.vecCons a (Matrix.vecCons b Matrix.vecEmpty) (Fin.castSucc 0)) …
  -/
  apply Fin.mk_le_mk.mpr hab
  /-
    🎉 no goals
  -/


lemma coe_edge_down_toOrderHom (n : ℕ) (a b : Fin (n+1)) (hab : a ≤ b) :
    ↑(edge n a b hab).down.toOrderHom = ![a, b] :=
  rfl


/-- The triangle in the standard simplex with vertices `a`, `b`, and `c`. -/
def triangle {n : ℕ} (a b c : Fin (n+1)) (hab : a ≤ b) (hbc : b ≤ c) : Δ[n] _[2] := by
  /-
    n : Nat
    a b c : Fin (HAdd.hAdd n 1)
    hab : LE.le a b
    hbc : LE.le b c
    ⊢ (SSet.standardSimplex.obj (SimplexCategory.mk n)).obj { unop := SimplexCateg …
  -/
  refine objMk ⟨![a, b, c], ?_⟩
  /-
    n : Nat
    a b c : Fin (HAdd.hAdd n 1)
    hab : LE.le a b
    hbc : LE.le b c
    ⊢ Monotone (Matrix.vecCons a (Matrix.vecCons b (Matrix.vecCons c Matrix.vecEmp …
  -/
  rw [Fin.monotone_iff_le_succ]
  /-
    n : Nat
    a b c : Fin (HAdd.hAdd n 1)
    hab : LE.le a b
    hbc : LE.le b c
    ⊢ ∀ (i : Fin (Opposite.unop { unop := SimplexCategory.mk 2 }).len), LE.le (Mat …
  -/
  simp only [unop_op, len_mk, Fin.forall_fin_two]
  /-
    n : Nat
    a b c : Fin (HAdd.hAdd n 1)
    hab : LE.le a b
    hbc : LE.le b c
    ⊢ And (LE.le (Matrix.vecCons a (Matrix.vecCons b (Matrix.vecCons c Matrix.vecE …
  -/
  dsimp
  /-
    n : Nat
    a b c : Fin (HAdd.hAdd n 1)
    hab : LE.le a b
    hbc : LE.le b c
    ⊢ And (LE.le a b) (LE.le b (Matrix.vecHead (Matrix.vecTail (Matrix.vecCons b ( …
  -/
  simp only [*, Matrix.tail_cons, Matrix.head_cons, true_and]
  /-
    🎉 no goals
  -/


lemma coe_triangle_down_toOrderHom {n : ℕ} (a b c : Fin (n+1)) (hab : a ≤ b) (hbc : b ≤ c) :
    ↑(triangle a b c hab hbc).down.toOrderHom = ![a, b, c] :=
  rfl


/-- The `m`-simplices of the `n`-th standard simplex are
the monotone maps from `Fin (m+1)` to `Fin (n+1)`. -/
def asOrderHom {n} {m} (α : Δ[n].obj m) : OrderHom (Fin (m.unop.len + 1)) (Fin (n + 1)) :=
  α.down.toOrderHom


/-- The boundary `∂Δ[n]` of the `n`-th standard simplex consists of
all `m`-simplices of `standardSimplex n` that are not surjective
(when viewed as monotone function `m → n`). -/
def boundary (n : ℕ) : SSet.{u} where
  obj m := { α : Δ[n].obj m // ¬Function.Surjective (asOrderHom α) }
  map {m₁ m₂} f α :=
    ⟨Δ[n].map f α.1, by
      /-
        n : Nat
        m₁ m₂ : Opposite SimplexCategory
        f : Quiver.Hom m₁ m₂
        α : (fun m => Subtype fun α => Not (Function.Surjective ⇑(SSet.asOrderHom α))) …
        ⊢ Not (Function.Surjective ⇑(SSet.asOrderHom ((SSet.standardSimplex.obj (Simpl …
      -/
      intro h
      /-
        n : Nat
        m₁ m₂ : Opposite SimplexCategory
        f : Quiver.Hom m₁ m₂
        α : (fun m => Subtype fun α => Not (Function.Surjective ⇑(SSet.asOrderHom α))) …
        h : Function.Surjective ⇑(SSet.asOrderHom ((SSet.standardSimplex.obj (SimplexC …
        ⊢ False
      -/
      apply α.property
      /-
        n : Nat
        m₁ m₂ : Opposite SimplexCategory
        f : Quiver.Hom m₁ m₂
        α : (fun m => Subtype fun α => Not (Function.Surjective ⇑(SSet.asOrderHom α))) …
        h : Function.Surjective ⇑(SSet.asOrderHom ((SSet.standardSimplex.obj (SimplexC …
        ⊢ Function.Surjective ⇑(SSet.asOrderHom ↑α)
      -/
      exact Function.Surjective.of_comp h⟩
      /-
        🎉 no goals
      -/


/-- The boundary `∂Δ[n]` of the `n`-th standard simplex -/
scoped[Simplicial] notation3 "∂Δ[" n "]" => SSet.boundary n


set_option linter.unusedVariables false in
/-- The inclusion of the boundary of the `n`-th standard simplex into that standard simplex. -/
def boundaryInclusion (n : ℕ) : ∂Δ[n] ⟶ Δ[n] where app m (α : { α : Δ[n].obj m // _ }) := α


/-- `horn n i` (or `Λ[n, i]`) is the `i`-th horn of the `n`-th standard simplex, where `i : n`.
It consists of all `m`-simplices `α` of `Δ[n]`
for which the union of `{i}` and the range of `α` is not all of `n`
(when viewing `α` as monotone function `m → n`). -/
def horn (n : ℕ) (i : Fin (n + 1)) : SSet where
  obj m := { α : Δ[n].obj m // Set.range (asOrderHom α) ∪ {i} ≠ Set.univ }
  map {m₁ m₂} f α :=
    ⟨Δ[n].map f α.1, by
      /-
        n : Nat
        i : Fin (HAdd.hAdd n 1)
        m₁ m₂ : Opposite SimplexCategory
        f : Quiver.Hom m₁ m₂
        α : (fun m => Subtype fun α => Ne (Union.union (Set.range ⇑(SSet.asOrderHom α) …
        ⊢ Ne (Union.union (Set.range ⇑(SSet.asOrderHom ((SSet.standardSimplex.obj (Sim …
      -/
      intro h; apply α.property
      /-
        n : Nat
        i : Fin (HAdd.hAdd n 1)
        m₁ m₂ : Opposite SimplexCategory
        f : Quiver.Hom m₁ m₂
        α : (fun m => Subtype fun α => Ne (Union.union (Set.range ⇑(SSet.asOrderHom α) …
        h : Eq (Union.union (Set.range ⇑(SSet.asOrderHom ((SSet.standardSimplex.obj (S …
        ⊢ Eq (Union.union (Set.range ⇑(SSet.asOrderHom ↑α)) (Singleton.singleton i)) S …
      -/
      rw [Set.eq_univ_iff_forall] at h ⊢; intro j
      /-
        n : Nat
        i : Fin (HAdd.hAdd n 1)
        m₁ m₂ : Opposite SimplexCategory
        f : Quiver.Hom m₁ m₂
        α : (fun m => Subtype fun α => Ne (Union.union (Set.range ⇑(SSet.asOrderHom α) …
        h : ∀ (x : Fin (HAdd.hAdd n 1)), Membership.mem (Union.union (Set.range ⇑(SSet …
        j : Fin (HAdd.hAdd n 1)
        ⊢ Membership.mem (Union.union (Set.range ⇑(SSet.asOrderHom ↑α)) (Singleton.sin …
      -/
      apply Or.imp _ id (h j)
      /-
        n : Nat
        i : Fin (HAdd.hAdd n 1)
        m₁ m₂ : Opposite SimplexCategory
        f : Quiver.Hom m₁ m₂
        α : (fun m => Subtype fun α => Ne (Union.union (Set.range ⇑(SSet.asOrderHom α) …
        h : ∀ (x : Fin (HAdd.hAdd n 1)), Membership.mem (Union.union (Set.range ⇑(SSet …
        j : Fin (HAdd.hAdd n 1)
        ⊢ Membership.mem (Set.range ⇑(SSet.asOrderHom ((SSet.standardSimplex.obj (Simp …
      -/
      intro hj
      /-
        n : Nat
        i : Fin (HAdd.hAdd n 1)
        m₁ m₂ : Opposite SimplexCategory
        f : Quiver.Hom m₁ m₂
        α : (fun m => Subtype fun α => Ne (Union.union (Set.range ⇑(SSet.asOrderHom α) …
        h : ∀ (x : Fin (HAdd.hAdd n 1)), Membership.mem (Union.union (Set.range ⇑(SSet …
        j : Fin (HAdd.hAdd n 1)
        hj : Membership.mem (Set.range ⇑(SSet.asOrderHom ((SSet.standardSimplex.obj (S …
        ⊢ Membership.mem (Set.range ⇑(SSet.asOrderHom ↑α)) j
      -/
      exact Set.range_comp_subset_range _ _ hj⟩
      /-
        🎉 no goals
      -/


/-- The `i`-th horn `Λ[n, i]` of the standard `n`-simplex -/
scoped[Simplicial] notation3 "Λ[" n ", " i "]" => SSet.horn (n : ℕ) i


set_option linter.unusedVariables false in
/-- The inclusion of the `i`-th horn of the `n`-th standard simplex into that standard simplex. -/
def hornInclusion (n : ℕ) (i : Fin (n + 1)) : Λ[n, i] ⟶ Δ[n] where
  app m (α : { α : Δ[n].obj m // _ }) := α


/-- The (degenerate) subsimplex of `Λ[n+2, i]` concentrated in vertex `k`. -/
@[simps]
def const (n : ℕ) (i k : Fin (n+3)) (m : SimplexCategoryᵒᵖ) : Λ[n+2, i].obj m := by
  /-
    n : Nat
    i k : Fin (HAdd.hAdd n 3)
    m : Opposite SimplexCategory
    ⊢ (SSet.horn (HAdd.hAdd n 2) i).obj m
  -/
  refine ⟨standardSimplex.const _ k _, ?_⟩
  suffices ¬ Finset.univ ⊆ {i, k} by
    simpa [← Set.univ_subset_iff, Set.subset_def, asOrderHom, not_or, Fin.forall_fin_one,
      subset_iff, mem_univ, @eq_comm _ _ k]
  /-
    n : Nat
    i k : Fin (HAdd.hAdd n 3)
    m : Opposite SimplexCategory
    ⊢ Not (HasSubset.Subset Finset.univ (Insert.insert i (Singleton.singleton k)))
  -/
  intro h
  /-
    n : Nat
    i k : Fin (HAdd.hAdd n 3)
    m : Opposite SimplexCategory
    h : HasSubset.Subset Finset.univ (Insert.insert i (Singleton.singleton k))
    ⊢ False
  -/
  have := (card_le_card h).trans card_le_two
  /-
    n : Nat
    i k : Fin (HAdd.hAdd n 3)
    m : Opposite SimplexCategory
    h : HasSubset.Subset Finset.univ (Insert.insert i (Singleton.singleton k))
    this : LE.le Finset.univ.card 2
    ⊢ False
  -/
  rw [card_fin] at this
  /-
    n : Nat
    i k : Fin (HAdd.hAdd n 3)
    m : Opposite SimplexCategory
    h : HasSubset.Subset Finset.univ (Insert.insert i (Singleton.singleton k))
    this : LE.le (HAdd.hAdd n 3) 2
    ⊢ False
  -/
  omega
  /-
    🎉 no goals
  -/


/-- The edge of `Λ[n, i]` with endpoints `a` and `b`.

This edge only exists if `{i, a, b}` has cardinality less than `n`. -/
@[simps]
def edge (n : ℕ) (i a b : Fin (n+1)) (hab : a ≤ b) (H : #{i, a, b} ≤ n) : Λ[n, i] _[1] := by
  /-
    n : Nat
    i a b : Fin (HAdd.hAdd n 1)
    hab : LE.le a b
    H : LE.le (Insert.insert i (Insert.insert a (Singleton.singleton b))).card n
    ⊢ (SSet.horn n i).obj { unop := SimplexCategory.mk 1 }
  -/
  refine ⟨standardSimplex.edge n a b hab, ?range⟩
  case range =>
    suffices ∃ x, ¬i = x ∧ ¬a = x ∧ ¬b = x by
      simpa only [unop_op, len_mk, Nat.reduceAdd, asOrderHom, yoneda_obj_obj, Set.union_singleton,
        ne_eq, ← Set.univ_subset_iff, Set.subset_def, Set.mem_univ, Set.mem_insert_iff,
        @eq_comm _ _ i, Set.mem_range, forall_const, not_forall, not_or, not_exists,
        Fin.forall_fin_two, Fin.isValue]
    contrapose! H
    replace H : univ ⊆ {i, a, b} :=
      fun x _ ↦ by simpa [or_iff_not_imp_left, eq_comm] using H x
    replace H := card_le_card H
    rwa [card_fin] at H


/-- Alternative constructor for the edge of `Λ[n, i]` with endpoints `a` and `b`,
assuming `3 ≤ n`. -/
@[simps!]
def edge₃ (n : ℕ) (i a b : Fin (n+1)) (hab : a ≤ b) (H : 3 ≤ n) :
    Λ[n, i] _[1] :=
  horn.edge n i a b hab <| Finset.card_le_three.trans H


/-- The edge of `Λ[n, i]` with endpoints `j` and `j+1`.

This constructor assumes `0 < i < n`,
which is the type of horn that occurs in the horn-filling condition of quasicategories. -/
@[simps!]
def primitiveEdge {n : ℕ} {i : Fin (n+1)}
    (h₀ : 0 < i) (hₙ : i < Fin.last n) (j : Fin n) :
    Λ[n, i] _[1] := by
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    h₀ : LT.lt 0 i
    hₙ : LT.lt i (Fin.last n)
    j : Fin n
    ⊢ (SSet.horn n i).obj { unop := SimplexCategory.mk 1 }
  -/
  refine horn.edge n i j.castSucc j.succ ?_ ?_
    /-
      case refine_1
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last n)
      j : Fin n
      ⊢ LE.le j.castSucc j.succ
    -/
  · simp only [← Fin.val_fin_le, Fin.coe_castSucc, Fin.val_succ, le_add_iff_nonneg_right, zero_le]
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    h₀ : LT.lt 0 i
    hₙ : LT.lt i (Fin.last n)
    j : Fin n
    ⊢ LE.le (Insert.insert i (Insert.insert j.castSucc (Singleton.singleton j.succ …
  -/
  simp only [← Fin.val_fin_lt, Fin.val_zero, Fin.val_last] at h₀ hₙ
  obtain rfl|hn : n = 2 ∨ 2 < n := by
    rw [eq_comm, or_comm, ← le_iff_lt_or_eq]; omega
    /-
      case refine_2.inl
      i : Fin (HAdd.hAdd 2 1)
      j : Fin 2
      h₀ : LT.lt 0 ↑i
      hₙ : LT.lt (↑i) 2
      ⊢ LE.le (Insert.insert i (Insert.insert j.castSucc (Singleton.singleton j.succ …
    -/
  · revert i j; decide
                /-
                  🎉 no goals
                -/
    /-
      case refine_2.inr
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      j : Fin n
      h₀ : LT.lt 0 ↑i
      hₙ : LT.lt (↑i) n
      hn : LT.lt 2 n
      ⊢ LE.le (Insert.insert i (Insert.insert j.castSucc (Singleton.singleton j.succ …
    -/
  · exact Finset.card_le_three.trans hn
    /-
      🎉 no goals
    -/


/-- The triangle in the standard simplex with vertices `k`, `k+1`, and `k+2`.

This constructor assumes `0 < i < n`,
which is the type of horn that occurs in the horn-filling condition of quasicategories. -/
@[simps]
def primitiveTriangle {n : ℕ} (i : Fin (n+4))
    (h₀ : 0 < i) (hₙ : i < Fin.last (n+3))
    (k : ℕ) (h : k < n+2) : Λ[n+3, i] _[2] := by
  refine ⟨standardSimplex.triangle
    (n := n+3) ⟨k, by omega⟩ ⟨k+1, by omega⟩ ⟨k+2, by omega⟩ ?_ ?_, ?_⟩
    /-
      case refine_1
      n : Nat
      i : Fin (HAdd.hAdd n 4)
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 3))
      k : Nat
      h : LT.lt k (HAdd.hAdd n 2)
      ⊢ LE.le ⟨k, ⋯⟩ ⟨HAdd.hAdd k 1, ⋯⟩
    -/
  · simp only [Fin.mk_le_mk, le_add_iff_nonneg_right, zero_le]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      n : Nat
      i : Fin (HAdd.hAdd n 4)
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 3))
      k : Nat
      h : LT.lt k (HAdd.hAdd n 2)
      ⊢ LE.le ⟨HAdd.hAdd k 1, ⋯⟩ ⟨HAdd.hAdd k 2, ⋯⟩
    -/
  · simp only [Fin.mk_le_mk, add_le_add_iff_left, one_le_two]
    /-
      🎉 no goals
    -/
  simp only [unop_op, SimplexCategory.len_mk, asOrderHom, SimplexCategory.Hom.toOrderHom_mk,
    OrderHom.const_coe_coe, Set.union_singleton, ne_eq, ← Set.univ_subset_iff, Set.subset_def,
    Set.mem_univ, Set.mem_insert_iff, Set.mem_range, Function.const_apply, exists_const,
    forall_true_left, not_forall, not_or, unop_op, not_exists,
    standardSimplex.triangle, OrderHom.coe_mk, @eq_comm _ _ i,
    standardSimplex.objMk, standardSimplex.objEquiv, Equiv.ulift]
  /-
    case refine_3
    n : Nat
    i : Fin (HAdd.hAdd n 4)
    h₀ : LT.lt 0 i
    hₙ : LT.lt i (Fin.last (HAdd.hAdd n 3))
    k : Nat
    h : LT.lt k (HAdd.hAdd n 2)
    ⊢ Exists fun x => And (Not (Eq i x)) (∀ (x_1 : Fin (HAdd.hAdd 2 1)), Not (Eq ( …
  -/
  dsimp
  /-
    case refine_3
    n : Nat
    i : Fin (HAdd.hAdd n 4)
    h₀ : LT.lt 0 i
    hₙ : LT.lt i (Fin.last (HAdd.hAdd n 3))
    k : Nat
    h : LT.lt k (HAdd.hAdd n 2)
    ⊢ Exists fun x => And (Not (Eq i x)) (∀ (x_1 : Fin 3), Not (Eq (Matrix.vecCons …
  -/
  by_cases hk0 : k = 0
    /-
      case pos
      n : Nat
      i : Fin (HAdd.hAdd n 4)
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 3))
      k : Nat
      h : LT.lt k (HAdd.hAdd n 2)
      hk0 : Eq k 0
      ⊢ Exists fun x => And (Not (Eq i x)) (∀ (x_1 : Fin 3), Not (Eq (Matrix.vecCons …
    -/
  · subst hk0
    /-
      case pos
      n : Nat
      i : Fin (HAdd.hAdd n 4)
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 3))
      h : LT.lt 0 (HAdd.hAdd n 2)
      ⊢ Exists fun x => And (Not (Eq i x)) (∀ (x_1 : Fin 3), Not (Eq (Matrix.vecCons …
    -/
    use Fin.last (n+3)
    /-
      case h
      n : Nat
      i : Fin (HAdd.hAdd n 4)
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 3))
      h : LT.lt 0 (HAdd.hAdd n 2)
      ⊢ And (Not (Eq i (Fin.last (HAdd.hAdd n 3)))) (∀ (x : Fin 3), Not (Eq (Matrix. …
    -/
    simp only [hₙ.ne, not_false_eq_true, Fin.zero_eta, zero_add, true_and]
    /-
      case h
      n : Nat
      i : Fin (HAdd.hAdd n 4)
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 3))
      h : LT.lt 0 (HAdd.hAdd n 2)
      ⊢ ∀ (x : Fin 3), Not (Eq (Matrix.vecCons 0 (Matrix.vecCons ⟨1, ⋯⟩ (Matrix.vecC …
    -/
    intro j
    /-
      case h
      n : Nat
      i : Fin (HAdd.hAdd n 4)
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 3))
      h : LT.lt 0 (HAdd.hAdd n 2)
      j : Fin 3
      ⊢ Not (Eq (Matrix.vecCons 0 (Matrix.vecCons ⟨1, ⋯⟩ (Matrix.vecCons ⟨2, ⋯⟩ Matr …
    -/
                    /-
                      🎉 no goals
                    -/
                    /-
                      🎉 no goals
                    -/
    fin_cases j <;> simp [Fin.ext_iff]
                    /-
                      🎉 no goals
                    -/
    /-
      case neg
      n : Nat
      i : Fin (HAdd.hAdd n 4)
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 3))
      k : Nat
      h : LT.lt k (HAdd.hAdd n 2)
      hk0 : Not (Eq k 0)
      ⊢ Exists fun x => And (Not (Eq i x)) (∀ (x_1 : Fin 3), Not (Eq (Matrix.vecCons …
    -/
  · use 0
    /-
      case h
      n : Nat
      i : Fin (HAdd.hAdd n 4)
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 3))
      k : Nat
      h : LT.lt k (HAdd.hAdd n 2)
      hk0 : Not (Eq k 0)
      ⊢ And (Not (Eq i 0)) (∀ (x : Fin 3), Not (Eq (Matrix.vecCons ⟨k, ⋯⟩ (Matrix.ve …
    -/
    simp only [h₀.ne', not_false_eq_true, true_and]
    /-
      case h
      n : Nat
      i : Fin (HAdd.hAdd n 4)
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 3))
      k : Nat
      h : LT.lt k (HAdd.hAdd n 2)
      hk0 : Not (Eq k 0)
      ⊢ ∀ (x : Fin 3), Not (Eq (Matrix.vecCons ⟨k, ⋯⟩ (Matrix.vecCons ⟨HAdd.hAdd k 1 …
    -/
    intro j
    /-
      case h
      n : Nat
      i : Fin (HAdd.hAdd n 4)
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 3))
      k : Nat
      h : LT.lt k (HAdd.hAdd n 2)
      hk0 : Not (Eq k 0)
      j : Fin 3
      ⊢ Not (Eq (Matrix.vecCons ⟨k, ⋯⟩ (Matrix.vecCons ⟨HAdd.hAdd k 1, ⋯⟩ (Matrix.ve …
    -/
                    /-
                      🎉 no goals
                    -/
                    /-
                      🎉 no goals
                    -/
    fin_cases j <;> simp [Fin.ext_iff, hk0]
                    /-
                      🎉 no goals
                    -/


/-- The `j`th subface of the `i`-th horn. -/
@[simps]
def face {n : ℕ} (i j : Fin (n+2)) (h : j ≠ i) : Λ[n+1, i] _[n] :=
  ⟨(standardSimplex.objEquiv _ _).symm (SimplexCategory.δ j), by
    simpa [← Set.univ_subset_iff, Set.subset_def, asOrderHom, SimplexCategory.δ, not_or,
      standardSimplex.objEquiv, asOrderHom, Equiv.ulift]⟩


/-- Two morphisms from a horn are equal if they are equal on all suitable faces. -/
protected
lemma hom_ext {n : ℕ} {i : Fin (n+2)} {S : SSet} (σ₁ σ₂ : Λ[n+1, i] ⟶ S)
    (h : ∀ (j) (h : j ≠ i), σ₁.app _ (face i j h) = σ₂.app _ (face i j h)) :
    σ₁ = σ₂ := by
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    S : SSet
    σ₁ σ₂ : Quiver.Hom (SSet.horn (HAdd.hAdd n 1) i) S
    h : ∀ (j : Fin (HAdd.hAdd n 2)) (h : Ne j i), Eq (σ₁.app { unop := SimplexCate …
    ⊢ Eq σ₁ σ₂
  -/
  apply NatTrans.ext; apply funext; apply Opposite.rec; apply SimplexCategory.rec
  /-
    case app.h.op.h
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    S : SSet
    σ₁ σ₂ : Quiver.Hom (SSet.horn (HAdd.hAdd n 1) i) S
    h : ∀ (j : Fin (HAdd.hAdd n 2)) (h : Ne j i), Eq (σ₁.app { unop := SimplexCate …
    ⊢ ∀ (n_1 : Nat), Eq (σ₁.app { unop := SimplexCategory.mk n_1 }) (σ₂.app { unop …
  -/
  intro m; ext f
  /-
    case app.h.op.h.h
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    S : SSet
    σ₁ σ₂ : Quiver.Hom (SSet.horn (HAdd.hAdd n 1) i) S
    h : ∀ (j : Fin (HAdd.hAdd n 2)) (h : Ne j i), Eq (σ₁.app { unop := SimplexCate …
    m : Nat
    f : (SSet.horn (HAdd.hAdd n 1) i).obj { unop := SimplexCategory.mk m }
    ⊢ Eq (σ₁.app { unop := SimplexCategory.mk m } f) (σ₂.app { unop := SimplexCate …
  -/
  obtain ⟨f', hf⟩ := (standardSimplex.objEquiv _ _).symm.surjective f.1
  obtain ⟨j, hji, hfj⟩ : ∃ j, ¬j = i ∧ ∀ k, f'.toOrderHom k ≠ j := by
    obtain ⟨f, hf'⟩ := f
    subst hf
    simpa [← Set.univ_subset_iff, Set.subset_def, asOrderHom, not_or] using hf'
  have H : f = (Λ[n+1, i].map (factor_δ f' j).op) (face i j hji) := by
    apply Subtype.ext
    apply (standardSimplex.objEquiv _ _).injective
    rw [← hf]
    exact (factor_δ_spec f' j hfj).symm
  /-
    case app.h.op.h.h.intro.intro.intro
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    S : SSet
    σ₁ σ₂ : Quiver.Hom (SSet.horn (HAdd.hAdd n 1) i) S
    h : ∀ (j : Fin (HAdd.hAdd n 2)) (h : Ne j i), Eq (σ₁.app { unop := SimplexCate …
    m : Nat
    f : (SSet.horn (HAdd.hAdd n 1) i).obj { unop := SimplexCategory.mk m }
    f' : Quiver.Hom (Opposite.unop { unop := SimplexCategory.mk m }) (SimplexCateg …
    hf : Eq ((SSet.standardSimplex.objEquiv (SimplexCategory.mk (HAdd.hAdd n 1)) { …
    j : Fin (HAdd.hAdd n 2)
    hji : Not (Eq j i)
    hfj : ∀ (k : Fin (HAdd.hAdd (Opposite.unop { unop := SimplexCategory.mk m }).l …
    H : Eq f ((SSet.horn (HAdd.hAdd n 1) i).map (SimplexCategory.factor_δ f' j).op …
    ⊢ Eq (σ₁.app { unop := SimplexCategory.mk m } f) (σ₂.app { unop := SimplexCate …
  -/
  have H₁ := congrFun (σ₁.naturality (factor_δ f' j).op) (face i j hji)
  /-
    case app.h.op.h.h.intro.intro.intro
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    S : SSet
    σ₁ σ₂ : Quiver.Hom (SSet.horn (HAdd.hAdd n 1) i) S
    h : ∀ (j : Fin (HAdd.hAdd n 2)) (h : Ne j i), Eq (σ₁.app { unop := SimplexCate …
    m : Nat
    f : (SSet.horn (HAdd.hAdd n 1) i).obj { unop := SimplexCategory.mk m }
    f' : Quiver.Hom (Opposite.unop { unop := SimplexCategory.mk m }) (SimplexCateg …
    hf : Eq ((SSet.standardSimplex.objEquiv (SimplexCategory.mk (HAdd.hAdd n 1)) { …
    j : Fin (HAdd.hAdd n 2)
    hji : Not (Eq j i)
    hfj : ∀ (k : Fin (HAdd.hAdd (Opposite.unop { unop := SimplexCategory.mk m }).l …
    H : Eq f ((SSet.horn (HAdd.hAdd n 1) i).map (SimplexCategory.factor_δ f' j).op …
    H₁ : Eq (CategoryTheory.CategoryStruct.comp ((SSet.horn (HAdd.hAdd n 1) i).map …
    ⊢ Eq (σ₁.app { unop := SimplexCategory.mk m } f) (σ₂.app { unop := SimplexCate …
  -/
  have H₂ := congrFun (σ₂.naturality (factor_δ f' j).op) (face i j hji)
  /-
    case app.h.op.h.h.intro.intro.intro
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    S : SSet
    σ₁ σ₂ : Quiver.Hom (SSet.horn (HAdd.hAdd n 1) i) S
    h : ∀ (j : Fin (HAdd.hAdd n 2)) (h : Ne j i), Eq (σ₁.app { unop := SimplexCate …
    m : Nat
    f : (SSet.horn (HAdd.hAdd n 1) i).obj { unop := SimplexCategory.mk m }
    f' : Quiver.Hom (Opposite.unop { unop := SimplexCategory.mk m }) (SimplexCateg …
    hf : Eq ((SSet.standardSimplex.objEquiv (SimplexCategory.mk (HAdd.hAdd n 1)) { …
    j : Fin (HAdd.hAdd n 2)
    hji : Not (Eq j i)
    hfj : ∀ (k : Fin (HAdd.hAdd (Opposite.unop { unop := SimplexCategory.mk m }).l …
    H : Eq f ((SSet.horn (HAdd.hAdd n 1) i).map (SimplexCategory.factor_δ f' j).op …
    H₁ : Eq (CategoryTheory.CategoryStruct.comp ((SSet.horn (HAdd.hAdd n 1) i).map …
    H₂ : Eq (CategoryTheory.CategoryStruct.comp ((SSet.horn (HAdd.hAdd n 1) i).map …
    ⊢ Eq (σ₁.app { unop := SimplexCategory.mk m } f) (σ₂.app { unop := SimplexCate …
  -/
  dsimp at H₁ H₂
  /-
    case app.h.op.h.h.intro.intro.intro
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    S : SSet
    σ₁ σ₂ : Quiver.Hom (SSet.horn (HAdd.hAdd n 1) i) S
    h : ∀ (j : Fin (HAdd.hAdd n 2)) (h : Ne j i), Eq (σ₁.app { unop := SimplexCate …
    m : Nat
    f : (SSet.horn (HAdd.hAdd n 1) i).obj { unop := SimplexCategory.mk m }
    f' : Quiver.Hom (Opposite.unop { unop := SimplexCategory.mk m }) (SimplexCateg …
    hf : Eq ((SSet.standardSimplex.objEquiv (SimplexCategory.mk (HAdd.hAdd n 1)) { …
    j : Fin (HAdd.hAdd n 2)
    hji : Not (Eq j i)
    hfj : ∀ (k : Fin (HAdd.hAdd (Opposite.unop { unop := SimplexCategory.mk m }).l …
    H : Eq f ((SSet.horn (HAdd.hAdd n 1) i).map (SimplexCategory.factor_δ f' j).op …
    H₁ : Eq (σ₁.app { unop := SimplexCategory.mk m } ((SSet.horn (HAdd.hAdd n 1) i …
    H₂ : Eq (σ₂.app { unop := SimplexCategory.mk m } ((SSet.horn (HAdd.hAdd n 1) i …
    ⊢ Eq (σ₁.app { unop := SimplexCategory.mk m } f) (σ₂.app { unop := SimplexCate …
  -/
  rw [H, H₁, H₂, h _ hji]
  /-
    🎉 no goals
  -/


/-- The simplicial circle. -/
noncomputable def S1 : SSet :=
  Limits.colimit <|
    Limits.parallelPair (standardSimplex.map <| SimplexCategory.δ 0 : Δ[0] ⟶ Δ[1])
      (standardSimplex.map <| SimplexCategory.δ 1)


/-- Truncated simplicial sets. -/
def Truncated (n : ℕ) :=
  SimplicialObject.Truncated (Type u) n


instance Truncated.largeCategory (n : ℕ) : LargeCategory (Truncated n) := by
  /-
    n : Nat
    ⊢ CategoryTheory.LargeCategory (SSet.Truncated n)
  -/
  dsimp only [Truncated]
  /-
    n : Nat
    ⊢ CategoryTheory.LargeCategory (CategoryTheory.SimplicialObject.Truncated (Typ …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance Truncated.hasLimits {n : ℕ} : HasLimits (Truncated n) := by
  /-
    n : Nat
    ⊢ CategoryTheory.Limits.HasLimits (SSet.Truncated n)
  -/
  dsimp only [Truncated]
  /-
    n : Nat
    ⊢ CategoryTheory.Limits.HasLimits (CategoryTheory.SimplicialObject.Truncated ( …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance Truncated.hasColimits {n : ℕ} : HasColimits (Truncated n) := by
  /-
    n : Nat
    ⊢ CategoryTheory.Limits.HasColimits (SSet.Truncated n)
  -/
  dsimp only [Truncated]
  /-
    n : Nat
    ⊢ CategoryTheory.Limits.HasColimits (CategoryTheory.SimplicialObject.Truncated …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The ulift functor `SSet.Truncated.{u} ⥤ SSet.Truncated.{max u v}` on truncated
simplicial sets. -/
def Truncated.uliftFunctor (k : ℕ) : SSet.Truncated.{u} k ⥤ SSet.Truncated.{max u v} k :=
  (whiskeringRight _ _ _).obj CategoryTheory.uliftFunctor.{v, u}


@[ext]
lemma Truncated.hom_ext {n : ℕ} {X Y : Truncated n} {f g : X ⟶ Y} (w : ∀ n, f.app n = g.app n) :
    f = g :=
  NatTrans.ext (funext w)


/-- The truncation functor on simplicial sets. -/
abbrev truncation (n : ℕ) : SSet ⥤ SSet.Truncated n := SimplicialObject.truncation n


instance {n} : Inhabited (SSet.Truncated n) :=
  ⟨(truncation n).obj <| Δ[0]⟩



/-- The n-skeleton as a functor `SSet.Truncated n ⥤ SSet`. -/
protected abbrev Truncated.sk (n : ℕ) : SSet.Truncated n ⥤ SSet.{u} :=
  SimplicialObject.Truncated.sk n


/-- The n-coskeleton as a functor `SSet.Truncated n ⥤ SSet`. -/
protected abbrev Truncated.cosk (n : ℕ) : SSet.Truncated n ⥤ SSet.{u} :=
  SimplicialObject.Truncated.cosk n


/-- The n-skeleton as an endofunctor on `SSet`. -/
abbrev sk (n : ℕ) : SSet.{u} ⥤ SSet.{u} := SimplicialObject.sk n


/-- The n-coskeleton as an endofunctor on `SSet`. -/
abbrev cosk (n : ℕ) : SSet.{u} ⥤ SSet.{u} := SimplicialObject.cosk n


/-- The adjunction between the n-skeleton and n-truncation.-/
noncomputable def skAdj (n : ℕ) : Truncated.sk n ⊣ truncation.{u} n :=
  SimplicialObject.skAdj n


/-- The adjunction between n-truncation and the n-coskeleton.-/
noncomputable def coskAdj (n : ℕ) : truncation.{u} n ⊣ Truncated.cosk n :=
  SimplicialObject.coskAdj n


instance cosk_reflective (n) : IsIso (coskAdj n).counit :=
  SimplicialObject.Truncated.cosk_reflective n


instance sk_coreflective (n) : IsIso (skAdj n).unit :=
  SimplicialObject.Truncated.sk_coreflective n


/-- Since `Truncated.inclusion` is fully faithful, so is right Kan extension along it.-/
noncomputable def cosk.fullyFaithful (n) :
    (Truncated.cosk n).FullyFaithful :=
  SimplicialObject.Truncated.cosk.fullyFaithful n


instance cosk.full (n) : (Truncated.cosk n).Full :=
  SimplicialObject.Truncated.cosk.full n


instance cosk.faithful (n) : (Truncated.cosk n).Faithful :=
  SimplicialObject.Truncated.cosk.faithful n


noncomputable instance coskAdj.reflective (n) : Reflective (Truncated.cosk n) :=
  SimplicialObject.Truncated.coskAdj.reflective n


/-- Since `Truncated.inclusion` is fully faithful, so is left Kan extension along it.-/
noncomputable def sk.fullyFaithful (n) :
    (Truncated.sk n).FullyFaithful := SimplicialObject.Truncated.sk.fullyFaithful n


instance sk.full (n) : (Truncated.sk n).Full := SimplicialObject.Truncated.sk.full n


instance sk.faithful (n) : (Truncated.sk n).Faithful :=
  SimplicialObject.Truncated.sk.faithful n


noncomputable instance skAdj.coreflective (n) : Coreflective (Truncated.sk n) :=
  SimplicialObject.Truncated.skAdj.coreflective n


/-- The category of augmented simplicial sets, as a particular case of
augmented simplicial objects. -/
abbrev Augmented :=
  SimplicialObject.Augmented (Type u)


/-- The functor which sends `[n]` to the simplicial set `Δ[n]` equipped by
the obvious augmentation towards the terminal object of the category of sets. -/
@[simps]
noncomputable def standardSimplex : SimplexCategory ⥤ SSet.Augmented.{u} where
  obj Δ :=
    { left := SSet.standardSimplex.obj Δ
      right := terminal _
      hom := { app := fun _ => terminal.from _ } }
  map θ :=
    { left := SSet.standardSimplex.map θ
      right := terminal.from _ }


lemma δ_comp_δ_apply {n} {i j : Fin (n + 2)} (H : i ≤ j) (x : S _[n + 2]) :
    S.δ i (S.δ j.succ x) = S.δ j (S.δ i.castSucc x) := congr_fun (S.δ_comp_δ H) x


lemma δ_comp_δ'_apply {n} {i : Fin (n + 2)} {j : Fin (n + 3)} (H : Fin.castSucc i < j)
    (x : S _[n + 2]) : S.δ i (S.δ j x) =
                                         /-
                                           S : SSet
                                           n : Nat
                                           i : Fin (HAdd.hAdd n 2)
                                           j : Fin (HAdd.hAdd n 3)
                                           H : LT.lt i.castSucc j
                                           x : S.obj { unop := SimplexCategory.mk (HAdd.hAdd n 2) }
                                           hj : Eq j 0
                                           ⊢ False
                                         -/
      S.δ (j.pred fun (hj : j = 0) => by simp [hj, Fin.not_lt_zero] at H) (S.δ i.castSucc x) :=
                                         /-
                                           🎉 no goals
                                         -/
  congr_fun (S.δ_comp_δ' H) x


lemma δ_comp_δ''_apply {n} {i : Fin (n + 3)} {j : Fin (n + 2)} (H : i ≤ Fin.castSucc j)
    (x : S _[n + 2]) :
    S.δ (i.castLT (Nat.lt_of_le_of_lt (Fin.le_iff_val_le_val.mp H) j.is_lt)) (S.δ j.succ x) =
      S.δ j (S.δ i x) := congr_fun (S.δ_comp_δ'' H) x


lemma δ_comp_δ_self_apply {n} {i : Fin (n + 2)} (x : S _[n + 2]) :
    S.δ i (S.δ i.castSucc x) = S.δ i (S.δ i.succ x) := congr_fun S.δ_comp_δ_self x


lemma δ_comp_δ_self'_apply {n} {i : Fin (n + 2)} {j : Fin (n + 3)} (H : j = Fin.castSucc i)
    (x : S _[n + 2]) : S.δ i (S.δ j x) = S.δ i (S.δ i.succ x) := congr_fun (S.δ_comp_δ_self' H) x


lemma δ_comp_σ_of_le_apply {n} {i : Fin (n + 2)} {j : Fin (n + 1)} (H : i ≤ Fin.castSucc j)
    (x : S _[n + 1]) :
    S.δ (Fin.castSucc i) (S.σ j.succ x) = S.σ j (S.δ i x) := congr_fun (S.δ_comp_σ_of_le H) x


@[simp]
lemma δ_comp_σ_self_apply {n} (i : Fin (n + 1)) (x : S _[n]) : S.δ i.castSucc (S.σ i x) = x :=
  congr_fun S.δ_comp_σ_self x


lemma δ_comp_σ_self'_apply {n} {j : Fin (n + 2)} {i : Fin (n + 1)} (H : j = Fin.castSucc i)
    (x : S _[n]) : S.δ j (S.σ i x) = x := congr_fun (S.δ_comp_σ_self' H) x


@[simp]
lemma δ_comp_σ_succ_apply {n} (i : Fin (n + 1)) (x : S _[n]) : S.δ i.succ (S.σ i x) = x :=
  congr_fun S.δ_comp_σ_succ x


lemma δ_comp_σ_succ'_apply {n} {j : Fin (n + 2)} {i : Fin (n + 1)} (H : j = i.succ) (x : S _[n]) :
    S.δ j (S.σ i x) = x := congr_fun (S.δ_comp_σ_succ' H) x


lemma δ_comp_σ_of_gt_apply {n} {i : Fin (n + 2)} {j : Fin (n + 1)} (H : Fin.castSucc j < i)
    (x : S _[n + 1]) : S.δ i.succ (S.σ (Fin.castSucc j) x) = S.σ j (S.δ i x) :=
  congr_fun (S.δ_comp_σ_of_gt H) x


lemma δ_comp_σ_of_gt'_apply {n} {i : Fin (n + 3)} {j : Fin (n + 2)} (H : j.succ < i)
    (x : S _[n + 1]) : S.δ i (S.σ j x) =
      S.σ (j.castLT ((add_lt_add_iff_right 1).mp (lt_of_lt_of_le H i.is_le)))
                                            /-
                                              S : SSet
                                              n : Nat
                                              i : Fin (HAdd.hAdd n 3)
                                              j : Fin (HAdd.hAdd n 2)
                                              H : LT.lt j.succ i
                                              x : S.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) }
                                              hi : Eq i 0
                                              ⊢ False
                                            -/
        (S.δ (i.pred fun (hi : i = 0) => by simp only [Fin.not_lt_zero, hi] at H) x) :=
                                            /-
                                              🎉 no goals
                                            -/
  congr_fun (S.δ_comp_σ_of_gt' H) x


lemma σ_comp_σ_apply {n} {i j : Fin (n + 1)} (H : i ≤ j) (x : S _[n]) :
    S.σ i.castSucc (S.σ j x) = S.σ j.succ (S.σ i x) := congr_fun (S.σ_comp_σ H) x


lemma δ_naturality_apply {n : ℕ} (i : Fin (n + 2)) (x : S _[n + 1]) :
    f.app (op [n]) (S.δ i x) = T.δ i (f.app (op [n + 1]) x) := by
  /-
    S T : SSet
    f : Quiver.Hom S T
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    x : S.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) }
    ⊢ Eq (f.app { unop := SimplexCategory.mk n } (CategoryTheory.SimplicialObject. …
  -/
  show (S.δ i ≫ f.app (op [n])) x = (f.app (op [n + 1]) ≫ T.δ i) x
  /-
    S T : SSet
    f : Quiver.Hom S T
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    x : S.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.SimplicialObject.δ S  …
  -/
  exact congr_fun (SimplicialObject.δ_naturality f i) x
  /-
    🎉 no goals
  -/


lemma σ_naturality_apply {n : ℕ} (i : Fin (n + 1)) (x : S _[n]) :
    f.app (op [n + 1]) (S.σ i x) = T.σ i (f.app (op [n]) x) := by
  /-
    S T : SSet
    f : Quiver.Hom S T
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    x : S.obj { unop := SimplexCategory.mk n }
    ⊢ Eq (f.app { unop := SimplexCategory.mk (HAdd.hAdd n 1) } (CategoryTheory.Sim …
  -/
  show (S.σ i ≫ f.app (op [n + 1])) x = (f.app (op [n]) ≫ T.σ i) x
  /-
    S T : SSet
    f : Quiver.Hom S T
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    x : S.obj { unop := SimplexCategory.mk n }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.SimplicialObject.σ S  …
  -/
  exact congr_fun (SimplicialObject.σ_naturality f i) x
  /-
    🎉 no goals
  -/


