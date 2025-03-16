/-- A cost structure for Levenshtein edit distance. -/
structure Cost (α β δ : Type*) where
  /-- Cost to delete an element from a list. -/
  delete : α → δ
  /-- Cost in insert an element into a list. -/
  insert : β → δ
  /-- Cost to substitute one element for another in a list. -/
  substitute : α → β → δ


/-- The default cost structure, for which all operations cost `1`. -/
@[simps]
def defaultCost [DecidableEq α] : Cost α α ℕ where
  delete _ := 1
  insert _ := 1
  substitute a b := if a = b then 0 else 1


instance [DecidableEq α] : Inhabited (Cost α α ℕ) := ⟨defaultCost⟩


/--
Cost structure given by a function.
Delete and insert cost the same, and substitution costs the greater value.
-/
@[simps]
def weightCost (f : α → ℕ) : Cost α α ℕ where
  delete a := f a
  insert b := f b
  substitute a b := max (f a) (f b)


/--
Cost structure for strings, where cost is the length of the token.
-/
@[simps!]
def stringLengthCost : Cost String String ℕ := weightCost String.length


/--
Cost structure for strings, where cost is the log base 2 length of the token.
-/
@[simps!]
def stringLogLengthCost : Cost String String ℕ := weightCost fun s => Nat.log2 (s.length + 1)


/--
(Implementation detail for `levenshtein`)

Given a list `xs` and the Levenshtein distances from each suffix of `xs` to some other list `ys`,
compute the Levenshtein distances from each suffix of `xs` to `y :: ys`.

(Note that we don't actually need to know `ys` itself here, so it is not an argument.)

The return value is a list of length `x.length + 1`,
and it is convenient for the recursive calls that we bundle this list
with a proof that it is non-empty.
-/
def impl
    (xs : List α) (y : β) (d : {r : List δ // 0 < r.length}) : {r : List δ // 0 < r.length} :=
  let ⟨ds, w⟩ := d
  xs.zip (ds.zip ds.tail) |>.foldr
                                                                   /-
                                                                     α : Type u_1
                                                                     β : Type u_2
                                                                     δ : Type u_3
                                                                     inst✝¹ : AddZeroClass δ
                                                                     inst✝ : Min δ
                                                                     C : Levenshtein.Cost α β δ
                                                                     xs : List α
                                                                     y : β
                                                                     d : Subtype fun r => LT.lt 0 r.length
                                                                     ds : List δ
                                                                     w : LT.lt 0 ds.length
                                                                     ⊢ LT.lt 0 (List.cons (HAdd.hAdd (C.insert y) (ds.getLast ⋯)) List.nil).length
                                                                   -/
    (init := ⟨[C.insert y + ds.getLast (List.length_pos.mp w)], by simp⟩)
                                                                                        /-
                                                                                          α : Type u_1
                                                                                          β : Type u_2
                                                                                          δ : Type u_3
                                                                                          inst✝¹ : AddZeroClass δ
                                                                                          inst✝ : Min δ
                                                                                          C : Levenshtein.Cost α β δ
                                                                                          xs : List α
                                                                                          y : β
                                                                                          d : Subtype fun r => LT.lt 0 r.length
                                                                                          ds : List δ
                                                                                          w✝ : LT.lt 0 ds.length
                                                                                          x✝¹ : Prod α (Prod δ δ)
                                                                                          x✝ : Subtype fun r => LT.lt 0 r.length
                                                                                          x : α
                                                                                          d₀ d₁ : δ
                                                                                          r : List δ
                                                                                          w : LT.lt 0 r.length
                                                                                          ⊢ LT.lt 0 (List.cons (Min.min (HAdd.hAdd (C.delete x) (GetElem.getElem r 0 w)) …
                                                                                        -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/
    (fun ⟨x, d₀, d₁⟩ ⟨r, w⟩ =>
      ⟨min (C.delete x + r[0]) (min (C.insert y + d₀) (C.substitute x y + d₁)) :: r, by simp⟩)


theorem impl_cons (w' : 0 < List.length ds) :
    impl C (x :: xs) y ⟨d :: ds, w⟩ =
      let ⟨r, w⟩ := impl C xs y ⟨ds, w'⟩
                                                                                          /-
                                                                                            α : Type u_1
                                                                                            β : Type u_2
                                                                                            δ : Type u_3
                                                                                            inst✝¹ : AddZeroClass δ
                                                                                            inst✝ : Min δ
                                                                                            C : Levenshtein.Cost α β δ
                                                                                            x : α
                                                                                            xs : List α
                                                                                            y : β
                                                                                            d : δ
                                                                                            ds : List δ
                                                                                            w✝ : LT.lt 0 (List.cons d ds).length
                                                                                            w' : LT.lt 0 ds.length
                                                                                            r : List δ
                                                                                            w : LT.lt 0 r.length
                                                                                            ⊢ LT.lt 0 (List.cons (Min.min (HAdd.hAdd (C.delete x) (GetElem.getElem r 0 w)) …
                                                                                          -/
      ⟨min (C.delete x + r[0]) (min (C.insert y + d) (C.substitute x y + ds[0])) :: r, by simp⟩ :=
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/
  match ds, w' with | _ :: _, _ => rfl

-- Note this lemma has two unspecified proofs: `h` appears on the left-hand-side
-- and should be found by matching, but `w'` will become an extra goal when rewriting.

theorem impl_cons_fst_zero (h : 0 < (impl C (x :: xs) y ⟨d :: ds, w⟩).val.length)
    (w' : 0 < List.length ds) : (impl C (x :: xs) y ⟨d :: ds, w⟩).1[0] =
      let ⟨r, w⟩ := impl C xs y ⟨ds, w'⟩
      min (C.delete x + r[0]) (min (C.insert y + d) (C.substitute x y + ds[0])) :=
  match ds, w' with | _ :: _, _ => rfl


theorem impl_length (d : {r : List δ // 0 < r.length}) (w : d.1.length = xs.length + 1) :
    (impl C xs y d).1.length = xs.length + 1 := by
  induction xs generalizing d with
  | nil => rfl
  | cons x xs ih =>
    dsimp [impl]
    match d, w with
    | ⟨d₁ :: d₂ :: ds, _⟩, w =>
      dsimp
      congr 1
      exact ih ⟨d₂ :: ds, (by simp)⟩ (by simpa using w)


/--
`suffixLevenshtein C xs ys` computes the Levenshtein distance
(using the cost functions provided by a `C : Cost α β δ`)
from each suffix of the list `xs` to the list `ys`.

The first element of this list is the Levenshtein distance from `xs` to `ys`.

Note that if the cost functions do not satisfy the inequalities
* `C.delete a + C.insert b ≥ C.substitute a b`
* `C.substitute a b + C.substitute b c ≥ C.substitute a c`
(or if any values are negative)
then the edit distances calculated here may not agree with the general
geodesic distance on the edit graph.
-/
def suffixLevenshtein (xs : List α) (ys : List β) : {r : List δ // 0 < r.length} :=
  ys.foldr
    (impl C xs)
                                                                                      /-
                                                                                        α : Type u_1
                                                                                        β : Type u_2
                                                                                        δ : Type u_3
                                                                                        inst✝¹ : AddZeroClass δ
                                                                                        inst✝ : Min δ
                                                                                        C : Levenshtein.Cost α β δ
                                                                                        xs : List α
                                                                                        ys : List β
                                                                                        a : α
                                                                                        x✝ : Subtype fun r => LT.lt 0 r.length
                                                                                        r : List δ
                                                                                        w : LT.lt 0 r.length
                                                                                        ⊢ LT.lt 0 (List.cons (HAdd.hAdd (C.delete a) (GetElem.getElem r 0 w)) r).length
                                                                                      -/
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
    (xs.foldr (init := ⟨[0], by simp⟩) (fun a ⟨r, w⟩ => ⟨(C.delete a + r[0]) :: r, by simp⟩))
                                /-
                                  🎉 no goals
                                -/


theorem suffixLevenshtein_length (xs : List α) (ys : List β) :
    (suffixLevenshtein C xs ys).1.length = xs.length + 1 := by
  induction ys with
  | nil =>
    dsimp [suffixLevenshtein]
    induction xs with
    | nil => rfl
    | cons _ xs ih =>
      simp_all
  | cons y ys ih =>
    dsimp [suffixLevenshtein]
    rw [impl_length]
    exact ih

-- This is only used in keeping track of estimates.

theorem suffixLevenshtein_eq (xs : List α) (y ys) :
    impl C xs y (suffixLevenshtein C xs ys) = suffixLevenshtein C xs (y :: ys) := by
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    inst✝¹ : AddZeroClass δ
    inst✝ : Min δ
    C : Levenshtein.Cost α β δ
    xs : List α
    y : β
    ys : List β
    ⊢ Eq (Levenshtein.impl C xs y (suffixLevenshtein C xs ys)) (suffixLevenshtein  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/--
`levenshtein C xs ys` computes the Levenshtein distance
(using the cost functions provided by a `C : Cost α β δ`)
from the list `xs` to the list `ys`.

Note that if the cost functions do not satisfy the inequalities
* `C.delete a + C.insert b ≥ C.substitute a b`
* `C.substitute a b + C.substitute b c ≥ C.substitute a c`
(or if any values are negative)
then the edit distance calculated here may not agree with the general
geodesic distance on the edit graph.
-/
def levenshtein (xs : List α) (ys : List β) : δ :=
  let ⟨r, w⟩ := suffixLevenshtein C xs ys
  r[0]


theorem suffixLevenshtein_nil_nil : (suffixLevenshtein C [] []).1 = [0] := by
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    inst✝¹ : AddZeroClass δ
    inst✝ : Min δ
    C : Levenshtein.Cost α β δ
    ⊢ Eq (suffixLevenshtein C List.nil List.nil).val (List.cons 0 List.nil)
  -/
  rfl
  /-
    🎉 no goals
  -/

-- Not sure if this belongs in the main `List` API, or can stay local.

theorem List.eq_of_length_one (x : List α) (w : x.length = 1) :
    have : 0 < x.length := lt_of_lt_of_eq Nat.zero_lt_one w.symm
    x = [x[0]] := by
  match x, w with
  | [r], _ => rfl


theorem suffixLevenshtein_nil' (ys : List β) :
    (suffixLevenshtein C [] ys).1 = [levenshtein C [] ys] :=
  List.eq_of_length_one _ (suffixLevenshtein_length [] _)


theorem suffixLevenshtein_cons₂ (xs : List α) (y ys) :
    suffixLevenshtein C xs (y :: ys) = (impl C xs) y (suffixLevenshtein C xs ys) :=
  rfl


theorem suffixLevenshtein_cons₁_aux {α} {x y : { l : List α // 0 < l.length }}
    (w₀ : x.1[0]'x.2 = y.1[0]'y.2) (w : x.1.tail = y.1.tail) : x = y := by
  match x, y with
  | ⟨hx :: tx, _⟩, ⟨hy :: ty, _⟩ => simp_all


theorem suffixLevenshtein_cons₁
    (x : α) (xs ys) :
    suffixLevenshtein C (x :: xs) ys =
      ⟨levenshtein C (x :: xs) ys ::
                                          /-
                                            α : Type u_1
                                            β : Type u_2
                                            δ : Type u_3
                                            inst✝¹ : AddZeroClass δ
                                            inst✝ : Min δ
                                            C : Levenshtein.Cost α β δ
                                            x : α
                                            xs : List α
                                            ys : List β
                                            ⊢ LT.lt 0 (List.cons (levenshtein C (List.cons x xs) ys) (suffixLevenshtein C  …
                                          -/
        (suffixLevenshtein C xs ys).1, by simp⟩ := by
                                          /-
                                            🎉 no goals
                                          -/
  induction ys with
  | nil =>
    dsimp [levenshtein, suffixLevenshtein]
  | cons y ys ih =>
    apply suffixLevenshtein_cons₁_aux
    · rfl
    · rw [suffixLevenshtein_cons₂ (x :: xs), ih, impl_cons]
      · rfl
      · simp [suffixLevenshtein_length]


theorem suffixLevenshtein_cons₁_fst (x : α) (xs ys) :
    (suffixLevenshtein C (x :: xs) ys).1 =
      levenshtein C (x :: xs) ys ::
        (suffixLevenshtein C xs ys).1 := by
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    inst✝¹ : AddZeroClass δ
    inst✝ : Min δ
    C : Levenshtein.Cost α β δ
    x : α
    xs : List α
    ys : List β
    ⊢ Eq (suffixLevenshtein C (List.cons x xs) ys).val (List.cons (levenshtein C ( …
  -/
  simp [suffixLevenshtein_cons₁]
  /-
    🎉 no goals
  -/


theorem suffixLevenshtein_cons_cons_fst_get_zero
    (x : α) (xs y ys) (w : 0 < (suffixLevenshtein C (x :: xs) (y :: ys)).val.length) :
    (suffixLevenshtein C (x :: xs) (y :: ys)).1[0]'w =
      let ⟨dx, _⟩ := suffixLevenshtein C xs (y :: ys)
      let ⟨dy, _⟩ := suffixLevenshtein C (x :: xs) ys
      let ⟨dxy, _⟩ := suffixLevenshtein C xs ys
      min
        (C.delete x + dx[0])
        (min
          (C.insert y + dy[0])
          (C.substitute x y + dxy[0])) := by
  conv =>
    lhs
    dsimp only [suffixLevenshtein_cons₂]
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    inst✝¹ : AddZeroClass δ
    inst✝ : Min δ
    C : Levenshtein.Cost α β δ
    x : α
    xs : List α
    y : β
    ys : List β
    w : LT.lt 0 (suffixLevenshtein C (List.cons x xs) (List.cons y ys)).val.length
    ⊢ Eq (GetElem.getElem (Levenshtein.impl C (List.cons x xs) y (suffixLevenshtei …
  -/
  simp only [suffixLevenshtein_cons₁]
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    inst✝¹ : AddZeroClass δ
    inst✝ : Min δ
    C : Levenshtein.Cost α β δ
    x : α
    xs : List α
    y : β
    ys : List β
    w : LT.lt 0 (suffixLevenshtein C (List.cons x xs) (List.cons y ys)).val.length
    ⊢ Eq (GetElem.getElem (Levenshtein.impl C (List.cons x xs) y ⟨List.cons (leven …
  -/
  rw [impl_cons_fst_zero]
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    inst✝¹ : AddZeroClass δ
    inst✝ : Min δ
    C : Levenshtein.Cost α β δ
    x : α
    xs : List α
    y : β
    ys : List β
    w : LT.lt 0 (suffixLevenshtein C (List.cons x xs) (List.cons y ys)).val.length
    ⊢ Eq (Levenshtein.impl.match_1 (fun x => δ) (Levenshtein.impl C xs y ⟨(suffixL …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem suffixLevenshtein_eq_tails_map (xs ys) :
    (suffixLevenshtein C xs ys).1 = xs.tails.map fun xs' => levenshtein C xs' ys := by
  induction xs with
  | nil =>
    simp only [suffixLevenshtein_nil', List.tails, List.map_cons, List.map]
  | cons x xs ih =>
    simp only [suffixLevenshtein_cons₁, ih, List.tails, List.map_cons]


@[simp]
theorem levenshtein_nil_nil : levenshtein C [] [] = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    inst✝¹ : AddZeroClass δ
    inst✝ : Min δ
    C : Levenshtein.Cost α β δ
    ⊢ Eq (levenshtein C List.nil List.nil) 0
  -/
  simp [levenshtein, suffixLevenshtein]
  /-
    🎉 no goals
  -/


@[simp]
theorem levenshtein_nil_cons (y) (ys) :
    levenshtein C [] (y :: ys) = C.insert y + levenshtein C [] ys := by
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    inst✝¹ : AddZeroClass δ
    inst✝ : Min δ
    C : Levenshtein.Cost α β δ
    y : β
    ys : List β
    ⊢ Eq (levenshtein C List.nil (List.cons y ys)) (HAdd.hAdd (C.insert y) (levens …
  -/
  dsimp (config := { unfoldPartialApp := true }) [levenshtein, suffixLevenshtein, impl]
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    inst✝¹ : AddZeroClass δ
    inst✝ : Min δ
    C : Levenshtein.Cost α β δ
    y : β
    ys : List β
    ⊢ Eq (HAdd.hAdd (C.insert y) ((List.foldr (fun y d => ⟨List.cons (HAdd.hAdd (C …
  -/
  congr
  /-
    case e_a
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    inst✝¹ : AddZeroClass δ
    inst✝ : Min δ
    C : Levenshtein.Cost α β δ
    y : β
    ys : List β
    ⊢ Eq ((List.foldr (fun y d => ⟨List.cons (HAdd.hAdd (C.insert y) (d.val.getLas …
  -/
  rw [List.getLast_eq_getElem]
  /-
    case e_a
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    inst✝¹ : AddZeroClass δ
    inst✝ : Min δ
    C : Levenshtein.Cost α β δ
    y : β
    ys : List β
    ⊢ Eq (GetElem.getElem (List.foldr (fun y d => ⟨List.cons (HAdd.hAdd (C.insert  …
  -/
  congr
  /-
    case e_a.e_i
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    inst✝¹ : AddZeroClass δ
    inst✝ : Min δ
    C : Levenshtein.Cost α β δ
    y : β
    ys : List β
    ⊢ Eq (HSub.hSub (List.foldr (fun y d => ⟨List.cons (HAdd.hAdd (C.insert y) (d. …
  -/
  rw [show (List.length _) = 1 from _]
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    inst✝¹ : AddZeroClass δ
    inst✝ : Min δ
    C : Levenshtein.Cost α β δ
    y : β
    ys : List β
    ⊢ Eq (List.foldr (fun y d => ⟨List.cons (HAdd.hAdd (C.insert y) (d.val.getLast …
  -/
                   /-
                     🎉 no goals
                   -/
  induction ys <;> simp
                   /-
                     🎉 no goals
                   -/


@[simp]
theorem levenshtein_cons_nil (x : α) (xs : List α) :
    levenshtein C (x :: xs) [] = C.delete x + levenshtein C xs [] :=
  rfl


@[simp]
theorem levenshtein_cons_cons
    (x : α) (xs : List α) (y : β) (ys : List β) :
    levenshtein C (x :: xs) (y :: ys) =
      min (C.delete x + levenshtein C xs (y :: ys))
        (min (C.insert y + levenshtein C (x :: xs) ys)
          (C.substitute x y + levenshtein C xs ys)) :=
  suffixLevenshtein_cons_cons_fst_get_zero ..

