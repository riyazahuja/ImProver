/-- Product of a list.

`List.prod [a, b, c] = ((1 * a) * b) * c` -/
@[to_additive existing]
def prod {α} [Mul α] [One α] : List α → α :=
  foldr (· * ·) 1


/-- The alternating sum of a list. -/
def alternatingSum {G : Type*} [Zero G] [Add G] [Neg G] : List G → G
  | [] => 0
  | g :: [] => g
  | g :: h :: t => g + -h + alternatingSum t


/-- The alternating product of a list. -/
@[to_additive existing]
def alternatingProd {G : Type*} [One G] [Mul G] [Inv G] : List G → G
  | [] => 1
  | g :: [] => g
  | g :: h :: t => g * h⁻¹ * alternatingProd t


@[to_additive existing, simp]
theorem prod_nil : ([] : List M).prod = 1 :=
  rfl


@[to_additive existing, simp]
theorem prod_cons {a} {l : List M} : (a :: l).prod = a * l.prod := rfl


@[to_additive]
lemma prod_induction
    (p : M → Prop) (hom : ∀ a b, p a → p b → p (a * b)) (unit : p 1) (base : ∀ x ∈ l, p x) :
    p l.prod := by
  induction l with
  | nil => simpa
  | cons a l ih =>
    rw [List.prod_cons]
    simp only [mem_cons, forall_eq_or_imp] at base
    exact hom _ _ (base.1) (ih base.2)


@[to_additive]
theorem prod_singleton : [a].prod = a :=
  mul_one a


@[to_additive]
theorem prod_one_cons : (1 :: l).prod = l.prod := by
  /-
    M : Type u_4
    inst✝ : MulOneClass M
    l : List M
    ⊢ Eq (List.cons 1 l).prod l.prod
  -/
  rw [prod, foldr, one_mul]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_map_one {l : List ι} :
    (l.map fun _ => (1 : M)).prod = 1 := by
  induction l with
  | nil => rfl
  | cons hd tl ih => rw [map_cons, prod_one_cons, ih]


@[to_additive]
theorem prod_eq_foldl : ∀ {l : List M}, l.prod = foldl (· * ·) 1 l
  | [] => rfl
  | cons a l => by
    /-
      M : Type u_4
      inst✝ : Monoid M
      a : M
      l : List M
      ⊢ Eq (List.cons a l).prod (List.foldl (fun x1 x2 => HMul.hMul x1 x2) 1 (List.c …
    -/
    rw [prod_cons, prod_eq_foldl, ← foldl_assoc (α := M) (op := (· * ·))]
    /-
      M : Type u_4
      inst✝ : Monoid M
      a : M
      l : List M
      ⊢ Eq (List.foldl (fun x1 x2 => HMul.hMul x1 x2) (HMul.hMul a 1) l) (List.foldl …
    -/
    simp
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp)]
theorem prod_append : (l₁ ++ l₂).prod = l₁.prod * l₂.prod :=
  calc
                                                                      /-
                                                                        M : Type u_4
                                                                        inst✝ : Monoid M
                                                                        l₁ l₂ : List M
                                                                        ⊢ Eq (HAppend.hAppend l₁ l₂).prod (List.foldr (fun x1 x2 => HMul.hMul x1 x2) ( …
                                                                      -/
    (l₁ ++ l₂).prod = foldr (· * ·) (1 * foldr (· * ·) 1 l₂) l₁ := by simp [List.prod]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
    _ = l₁.prod * l₂.prod := foldr_assoc


@[to_additive]
theorem prod_concat : (l.concat a).prod = l.prod * a := by
  /-
    M : Type u_4
    inst✝ : Monoid M
    l : List M
    a : M
    ⊢ Eq (l.concat a).prod (HMul.hMul l.prod a)
  -/
  rw [concat_eq_append, prod_append, prod_singleton]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem prod_flatten {l : List (List M)} : l.flatten.prod = (l.map List.prod).prod := by
  induction l with
  | nil => simp
  | cons head tail ih => simp only [*, List.flatten, map, prod_append, prod_cons]


@[deprecated (since := "2024-10-15")] alias prod_join := prod_flatten

@[deprecated (since := "2024-10-15")] alias sum_join := sum_flatten


@[to_additive]
theorem prod_eq_foldr {l : List M} : l.prod = foldr (· * ·) 1 l := rfl


@[to_additive (attr := simp)]
theorem prod_replicate (n : ℕ) (a : M) : (replicate n a).prod = a ^ n := by
  induction n with
  | zero => rw [pow_zero, replicate_zero, prod_nil]
  | succ n ih => rw [replicate_succ, prod_cons, ih, pow_succ']


@[to_additive sum_eq_card_nsmul]
theorem prod_eq_pow_card (l : List M) (m : M) (h : ∀ x ∈ l, x = m) : l.prod = m ^ l.length := by
  /-
    M : Type u_4
    inst✝ : Monoid M
    l : List M
    m : M
    h : ∀ (x : M), Membership.mem l x → Eq x m
    ⊢ Eq l.prod (HPow.hPow m l.length)
  -/
  rw [← prod_replicate, ← List.eq_replicate_iff.mpr ⟨rfl, h⟩]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_hom_rel (l : List ι) {r : M → N → Prop} {f : ι → M} {g : ι → N} (h₁ : r 1 1)
    (h₂ : ∀ ⦃i a b⦄, r a b → r (f i * a) (g i * b)) : r (l.map f).prod (l.map g).prod :=
                                   /-
                                     ι : Type u_1
                                     M : Type u_4
                                     N : Type u_5
                                     inst✝¹ : Monoid M
                                     inst✝ : Monoid N
                                     l✝ : List ι
                                     r : M → N → Prop
                                     f : ι → M
                                     g : ι → N
                                     h₁ : r 1 1
                                     h₂ : ∀ ⦃i : ι⦄ ⦃a : M⦄ ⦃b : N⦄, r a b → r (HMul.hMul (f i) a) (HMul.hMul (g i) …
                                     a : ι
                                     l : List ι
                                     hl : r (List.map f l).prod (List.map g l).prod
                                     ⊢ r (List.map f (List.cons a l)).prod (List.map g (List.cons a l)).prod
                                   -/
  List.recOn l h₁ fun a l hl => by simp only [map_cons, prod_cons, h₂ hl]
                                   /-
                                     🎉 no goals
                                   -/


@[to_additive]
theorem rel_prod {R : M → N → Prop} (h : R 1 1) (hf : (R ⇒ R ⇒ R) (· * ·) (· * ·)) :
    (Forall₂ R ⇒ R) prod prod :=
  rel_foldr hf h


@[to_additive]
theorem prod_hom_nonempty {l : List M} {F : Type*} [FunLike F M N] [MulHomClass F M N] (f : F)
    (hl : l ≠ []) : (l.map f).prod = f l.prod :=
                                       /-
                                         M : Type u_4
                                         N : Type u_5
                                         inst✝³ : Monoid M
                                         inst✝² : Monoid N
                                         l : List M
                                         F : Type u_8
                                         inst✝¹ : FunLike F M N
                                         inst✝ : MulHomClass F M N
                                         f : F
                                         hl✝ : Ne l List.nil
                                         x : M
                                         xs : List M
                                         hl : Ne (List.cons x xs) List.nil
                                         ⊢ Eq (List.map (⇑f) (List.cons x xs)).prod (f (List.cons x xs).prod)
                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
  match l, hl with | x :: xs, hl => by induction xs generalizing x <;> aesop
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[to_additive]
theorem prod_hom (l : List M) {F : Type*} [FunLike F M N] [MonoidHomClass F M N] (f : F) :
    (l.map f).prod = f l.prod := by
  /-
    M : Type u_4
    N : Type u_5
    inst✝³ : Monoid M
    inst✝² : Monoid N
    l : List M
    F : Type u_8
    inst✝¹ : FunLike F M N
    inst✝ : MonoidHomClass F M N
    f : F
    ⊢ Eq (List.map (⇑f) l).prod (f l.prod)
  -/
  simp only [prod, foldr_map, ← map_one f]
  /-
    M : Type u_4
    N : Type u_5
    inst✝³ : Monoid M
    inst✝² : Monoid N
    l : List M
    F : Type u_8
    inst✝¹ : FunLike F M N
    inst✝ : MonoidHomClass F M N
    f : F
    ⊢ Eq (List.foldr (fun x y => HMul.hMul (f x) y) (f 1) l) (f (List.foldr (fun x …
  -/
  exact l.foldr_hom f (· * ·) (f · * ·) 1 (fun x y => (map_mul f x y).symm)
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_hom₂_nonempty {l : List ι} (f : M → N → P)
    (hf : ∀ a b c d, f (a * b) (c * d) = f a c * f b d) (f₁ : ι → M) (f₂ : ι → N) (hl : l ≠ []) :
    (l.map fun i => f (f₁ i) (f₂ i)).prod = f (l.map f₁).prod (l.map f₂).prod := by
  /-
    ι : Type u_1
    M : Type u_4
    N : Type u_5
    P : Type u_6
    inst✝² : Monoid M
    inst✝¹ : Monoid N
    inst✝ : Monoid P
    l : List ι
    f : M → N → P
    hf : ∀ (a b : M) (c d : N), Eq (f (HMul.hMul a b) (HMul.hMul c d)) (HMul.hMul  …
    f₁ : ι → M
    f₂ : ι → N
    hl : Ne l List.nil
    ⊢ Eq (List.map (fun i => f (f₁ i) (f₂ i)) l).prod (f (List.map f₁ l).prod (Lis …
  -/
  match l, hl with | x :: xs, hl => induction xs generalizing x <;> aesop
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_hom₂ (l : List ι) (f : M → N → P) (hf : ∀ a b c d, f (a * b) (c * d) = f a c * f b d)
    (hf' : f 1 1 = 1) (f₁ : ι → M) (f₂ : ι → N) :
    (l.map fun i => f (f₁ i) (f₂ i)).prod = f (l.map f₁).prod (l.map f₂).prod := by
  rw [prod, prod, prod, foldr_map, foldr_map, foldr_map,
    ← l.foldr_hom₂ f _ _ (fun x y => f (f₁ x) (f₂ x) * y) _ _ (by simp [hf]), hf']


@[to_additive (attr := simp)]
theorem prod_map_mul {α : Type*} [CommMonoid α] {l : List ι} {f g : ι → α} :
    (l.map fun i => f i * g i).prod = (l.map f).prod * (l.map g).prod :=
  l.prod_hom₂ (· * ·) mul_mul_mul_comm (mul_one _) _ _


@[to_additive]
theorem prod_map_hom (L : List ι) (f : ι → M) {G : Type*} [FunLike G M N] [MonoidHomClass G M N]
    (g : G) :
                                                  /-
                                                    ι : Type u_1
                                                    M : Type u_4
                                                    N : Type u_5
                                                    inst✝³ : Monoid M
                                                    inst✝² : Monoid N
                                                    L : List ι
                                                    f : ι → M
                                                    G : Type u_8
                                                    inst✝¹ : FunLike G M N
                                                    inst✝ : MonoidHomClass G M N
                                                    g : G
                                                    ⊢ Eq (List.map (Function.comp (⇑g) f) L).prod (g (List.map f L).prod)
                                                  -/
    (L.map (g ∘ f)).prod = g (L.map f).prod := by rw [← prod_hom, map_map]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[to_additive]
theorem prod_isUnit : ∀ {L : List M}, (∀ m ∈ L, IsUnit m) → IsUnit L.prod
                /-
                  M : Type u_4
                  inst✝ : Monoid M
                  x✝ : ∀ (m : M), Membership.mem List.nil m → IsUnit m
                  ⊢ IsUnit List.nil.prod
                -/
  | [], _ => by simp
                /-
                  🎉 no goals
                -/
  | h :: t, u => by
    /-
      M : Type u_4
      inst✝ : Monoid M
      h : M
      t : List M
      u : ∀ (m : M), Membership.mem (List.cons h t) m → IsUnit m
      ⊢ IsUnit (List.cons h t).prod
    -/
    simp only [List.prod_cons]
    /-
      M : Type u_4
      inst✝ : Monoid M
      h : M
      t : List M
      u : ∀ (m : M), Membership.mem (List.cons h t) m → IsUnit m
      ⊢ IsUnit (HMul.hMul h t.prod)
    -/
    exact IsUnit.mul (u h (mem_cons_self h t)) (prod_isUnit fun m mt => u m (mem_cons_of_mem h mt))
    /-
      🎉 no goals
    -/


@[to_additive]
theorem prod_isUnit_iff {α : Type*} [CommMonoid α] {L : List α} :
    IsUnit L.prod ↔ ∀ m ∈ L, IsUnit m := by
  /-
    α : Type u_8
    inst✝ : CommMonoid α
    L : List α
    ⊢ Iff (IsUnit L.prod) (∀ (m : α), Membership.mem L m → IsUnit m)
  -/
  refine ⟨fun h => ?_, prod_isUnit⟩
  induction L with
  | nil => exact fun m' h' => False.elim (not_mem_nil m' h')
  | cons m L ih =>
    rw [prod_cons, IsUnit.mul_iff] at h
    exact fun m' h' ↦ Or.elim (eq_or_mem_of_mem_cons h') (fun H => H.substr h.1) fun H => ih h.2 _ H


@[to_additive (attr := simp)]
theorem prod_take_mul_prod_drop (L : List M) (i : ℕ) :
    (L.take i).prod * (L.drop i).prod = L.prod := by
  /-
    M : Type u_4
    inst✝ : Monoid M
    L : List M
    i : Nat
    ⊢ Eq (HMul.hMul (List.take i L).prod (List.drop i L).prod) L.prod
  -/
  simp [← prod_append]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem prod_take_succ (L : List M) (i : ℕ) (p : i < L.length) :
    (L.take (i + 1)).prod = (L.take i).prod * L[i] := by
  /-
    M : Type u_4
    inst✝ : Monoid M
    L : List M
    i : Nat
    p : LT.lt i L.length
    ⊢ Eq (List.take (HAdd.hAdd i 1) L).prod (HMul.hMul (List.take i L).prod (GetEl …
  -/
  rw [← take_concat_get' _ _ p, prod_append]
  /-
    M : Type u_4
    inst✝ : Monoid M
    L : List M
    i : Nat
    p : LT.lt i L.length
    ⊢ Eq (HMul.hMul (List.take i L).prod (List.cons (GetElem.getElem L i p) List.n …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- A list with product not one must have positive length. -/
@[to_additive "A list with sum not zero must have positive length."]
theorem length_pos_of_prod_ne_one (L : List M) (h : L.prod ≠ 1) : 0 < L.length := by
  /-
    M : Type u_4
    inst✝ : Monoid M
    L : List M
    h : Ne L.prod 1
    ⊢ LT.lt 0 L.length
  -/
  cases L
    /-
      case nil
      M : Type u_4
      inst✝ : Monoid M
      h : Ne List.nil.prod 1
      ⊢ LT.lt 0 List.nil.length
    -/
  · simp at h
    /-
      🎉 no goals
    -/
    /-
      case cons
      M : Type u_4
      inst✝ : Monoid M
      head✝ : M
      tail✝ : List M
      h : Ne (List.cons head✝ tail✝).prod 1
      ⊢ LT.lt 0 (List.cons head✝ tail✝).length
    -/
  · simp
    /-
      🎉 no goals
    -/


/-- A list with product greater than one must have positive length. -/
@[to_additive length_pos_of_sum_pos "A list with positive sum must have positive length."]
theorem length_pos_of_one_lt_prod [Preorder M] (L : List M) (h : 1 < L.prod) : 0 < L.length :=
  length_pos_of_prod_ne_one L h.ne'


/-- A list with product less than one must have positive length. -/
@[to_additive "A list with negative sum must have positive length."]
theorem length_pos_of_prod_lt_one [Preorder M] (L : List M) (h : L.prod < 1) : 0 < L.length :=
  length_pos_of_prod_ne_one L h.ne


@[to_additive]
theorem prod_set :
    ∀ (L : List M) (n : ℕ) (a : M),
      (L.set n a).prod =
        ((L.take n).prod * if n < L.length then a else 1) * (L.drop (n + 1)).prod
                        /-
                          M : Type u_4
                          inst✝ : Monoid M
                          x : M
                          xs : List M
                          a : M
                          ⊢ Eq ((List.cons x xs).set 0 a).prod (HMul.hMul (HMul.hMul (List.take 0 (List. …
                        -/
  | x :: xs, 0, a => by simp [set]
                        /-
                          🎉 no goals
                        -/
  | x :: xs, i + 1, a => by
    /-
      M : Type u_4
      inst✝ : Monoid M
      x : M
      xs : List M
      i : Nat
      a : M
      ⊢ Eq ((List.cons x xs).set (HAdd.hAdd i 1) a).prod (HMul.hMul (HMul.hMul (List …
    -/
    simp [set, prod_set xs i a, mul_assoc, Nat.add_lt_add_iff_right]
    /-
      🎉 no goals
    -/
                   /-
                     M : Type u_4
                     inst✝ : Monoid M
                     x✝¹ : Nat
                     x✝ : M
                     ⊢ Eq (List.nil.set x✝¹ x✝).prod (HMul.hMul (HMul.hMul (List.take x✝¹ List.nil) …
                   -/
  | [], _, _ => by simp [set, (Nat.zero_le _).not_lt, Nat.zero_le]
                   /-
                     🎉 no goals
                   -/


/-- We'd like to state this as `L.headI * L.tail.prod = L.prod`, but because `L.headI` relies on an
inhabited instance to return a garbage value on the empty list, this is not possible.
Instead, we write the statement in terms of `(L.get? 0).getD 1`.
-/
@[to_additive "We'd like to state this as `L.headI + L.tail.sum = L.sum`, but because `L.headI`
  relies on an inhabited instance to return a garbage value on the empty list, this is not possible.
  Instead, we write the statement in terms of `(L.get? 0).getD 0`."]
theorem get?_zero_mul_tail_prod (l : List M) : (l.get? 0).getD 1 * l.tail.prod = l.prod := by
  /-
    M : Type u_4
    inst✝ : Monoid M
    l : List M
    ⊢ Eq (HMul.hMul ((l.get? 0).getD 1) l.tail.prod) l.prod
  -/
              /-
                🎉 no goals
              -/
  cases l <;> simp
              /-
                🎉 no goals
              -/


/-- Same as `get?_zero_mul_tail_prod`, but avoiding the `List.headI` garbage complication by
  requiring the list to be nonempty. -/
@[to_additive "Same as `get?_zero_add_tail_sum`, but avoiding the `List.headI` garbage complication
  by requiring the list to be nonempty."]
theorem headI_mul_tail_prod_of_ne_nil [Inhabited M] (l : List M) (h : l ≠ []) :
                                         /-
                                           M : Type u_4
                                           inst✝¹ : Monoid M
                                           inst✝ : Inhabited M
                                           l : List M
                                           h : Ne l List.nil
                                           ⊢ Eq (HMul.hMul l.headI l.tail.prod) l.prod
                                         -/
    l.headI * l.tail.prod = l.prod := by cases l <;> [contradiction; simp]
                                         /-
                                           🎉 no goals
                                         -/


@[to_additive]
theorem _root_.Commute.list_prod_right (l : List M) (y : M) (h : ∀ x ∈ l, Commute y x) :
    Commute y l.prod := by
  induction l with
  | nil => simp
  | cons z l IH =>
    rw [List.forall_mem_cons] at h
    rw [List.prod_cons]
    exact Commute.mul_right h.1 (IH h.2)


@[to_additive]
theorem _root_.Commute.list_prod_left (l : List M) (y : M) (h : ∀ x ∈ l, Commute x y) :
    Commute l.prod y :=
  ((Commute.list_prod_right _ _) fun _ hx => (h _ hx).symm).symm


@[to_additive] lemma prod_range_succ (f : ℕ → M) (n : ℕ) :
    ((range n.succ).map f).prod = ((range n).map f).prod * f n := by
  /-
    M : Type u_4
    inst✝ : Monoid M
    f : Nat → M
    n : Nat
    ⊢ Eq (List.map f (List.range n.succ)).prod (HMul.hMul (List.map f (List.range  …
  -/
  rw [range_succ, map_append, map_singleton, prod_append, prod_cons, prod_nil, mul_one]
  /-
    🎉 no goals
  -/


/-- A variant of `prod_range_succ` which pulls off the first term in the product rather than the
last. -/
@[to_additive
"A variant of `sum_range_succ` which pulls off the first term in the sum rather than the last."]
lemma prod_range_succ' (f : ℕ → M) (n : ℕ) :
    ((range n.succ).map f).prod = f 0 * ((range n).map fun i ↦ f i.succ).prod := by
  /-
    M : Type u_4
    inst✝ : Monoid M
    f : Nat → M
    n : Nat
    ⊢ Eq (List.map f (List.range n.succ)).prod (HMul.hMul (f 0) (List.map (fun i = …
  -/
  rw [range_succ_eq_map]
  /-
    M : Type u_4
    inst✝ : Monoid M
    f : Nat → M
    n : Nat
    ⊢ Eq (List.map f (List.cons 0 (List.map Nat.succ (List.range n)))).prod (HMul. …
  -/
  simp [Function.comp_def]
  /-
    🎉 no goals
  -/


@[to_additive] lemma prod_eq_one (hl : ∀ x ∈ l, x = 1) : l.prod = 1 := by
  induction l with
  | nil => rfl
  | cons i l hil =>
    rw [List.prod_cons, hil fun x hx ↦ hl _ (mem_cons_of_mem i hx),
      hl _ (mem_cons_self i l), one_mul]


@[to_additive] lemma exists_mem_ne_one_of_prod_ne_one (h : l.prod ≠ 1) :
                               /-
                                 M : Type u_4
                                 inst✝ : Monoid M
                                 l : List M
                                 h : Ne l.prod 1
                                 ⊢ Exists fun x => And (Membership.mem l x) (Ne x 1)
                               -/
    ∃ x ∈ l, x ≠ (1 : M) := by simpa only [not_forall, exists_prop] using mt prod_eq_one h
                               /-
                                 🎉 no goals
                               -/


@[to_additive]
lemma prod_erase_of_comm [DecidableEq M] (ha : a ∈ l) (comm : ∀ x ∈ l, ∀ y ∈ l, x * y = y * x) :
    a * (l.erase a).prod = l.prod := by
  induction l with
  | nil => simp only [not_mem_nil] at ha
  | cons b l ih =>
    obtain rfl | ⟨ne, h⟩ := List.eq_or_ne_mem_of_mem ha
    · simp only [erase_cons_head, prod_cons]
    rw [List.erase, beq_false_of_ne ne.symm, List.prod_cons, List.prod_cons, ← mul_assoc,
      comm a ha b (l.mem_cons_self b), mul_assoc,
      ih h fun x hx y hy ↦ comm _ (List.mem_cons_of_mem b hx) _ (List.mem_cons_of_mem b hy)]


@[to_additive]
lemma prod_map_eq_pow_single [DecidableEq α] {l : List α} (a : α) (f : α → M)
    (hf : ∀ a', a' ≠ a → a' ∈ l → f a' = 1) : (l.map f).prod = f a ^ l.count a := by
  induction l generalizing a with
  | nil => rw [map_nil, prod_nil, count_nil, _root_.pow_zero]
  | cons a' as h =>
    specialize h a fun a' ha' hfa' => hf a' ha' (mem_cons_of_mem _ hfa')
    rw [List.map_cons, List.prod_cons, count_cons, h]
    simp only [beq_iff_eq]
    split_ifs with ha'
    · rw [ha', _root_.pow_succ']
    · rw [hf a' ha' (List.mem_cons_self a' as), one_mul, add_zero]


@[to_additive]
lemma prod_eq_pow_single [DecidableEq M] (a : M) (h : ∀ a', a' ≠ a → a' ∈ l → a' = 1) :
    l.prod = a ^ l.count a :=
                   /-
                     M : Type u_4
                     inst✝¹ : Monoid M
                     l : List M
                     inst✝ : DecidableEq M
                     a : M
                     h : ∀ (a' : M), Ne a' a → Membership.mem l a' → Eq a' 1
                     ⊢ Eq l.prod (List.map id l).prod
                   -/
  _root_.trans (by rw [map_id]) (prod_map_eq_pow_single a id h)
                   /-
                     🎉 no goals
                   -/


/-- If elements of a list commute with each other, then their product does not
depend on the order of elements. -/
@[to_additive "If elements of a list additively commute with each other, then their sum does not
depend on the order of elements."]
lemma Perm.prod_eq' (h : l₁ ~ l₂) (hc : l₁.Pairwise Commute) : l₁.prod = l₂.prod := by
  /-
    M : Type u_4
    inst✝ : Monoid M
    l₁ l₂ : List M
    h : l₁.Perm l₂
    hc : List.Pairwise Commute l₁
    ⊢ Eq l₁.prod l₂.prod
  -/
  refine h.foldr_eq' ?_ _
  /-
    M : Type u_4
    inst✝ : Monoid M
    l₁ l₂ : List M
    h : l₁.Perm l₂
    hc : List.Pairwise Commute l₁
    ⊢ ∀ (x : M), Membership.mem l₁ x → ∀ (y : M), Membership.mem l₁ y → ∀ (z : M), …
  -/
  apply Pairwise.forall_of_forall
    /-
      case H
      M : Type u_4
      inst✝ : Monoid M
      l₁ l₂ : List M
      h : l₁.Perm l₂
      hc : List.Pairwise Commute l₁
      ⊢ Symmetric fun x y => ∀ (z : M), Eq (HMul.hMul y (HMul.hMul x z)) (HMul.hMul  …
    -/
  · intro x y h z
    /-
      case H
      M : Type u_4
      inst✝ : Monoid M
      l₁ l₂ : List M
      h✝ : l₁.Perm l₂
      hc : List.Pairwise Commute l₁
      x y : M
      h : ∀ (z : M), Eq (HMul.hMul y (HMul.hMul x z)) (HMul.hMul x (HMul.hMul y z))
      z : M
      ⊢ Eq (HMul.hMul x (HMul.hMul y z)) (HMul.hMul y (HMul.hMul x z))
    -/
    exact (h z).symm
    /-
      🎉 no goals
    -/
    /-
      case H₁
      M : Type u_4
      inst✝ : Monoid M
      l₁ l₂ : List M
      h : l₁.Perm l₂
      hc : List.Pairwise Commute l₁
      ⊢ ∀ (x : M), Membership.mem l₁ x → ∀ (z : M), Eq (HMul.hMul x (HMul.hMul x z)) …
    -/
  · intros; rfl
            /-
              🎉 no goals
            -/
    /-
      case H₂
      M : Type u_4
      inst✝ : Monoid M
      l₁ l₂ : List M
      h : l₁.Perm l₂
      hc : List.Pairwise Commute l₁
      ⊢ List.Pairwise (fun x y => ∀ (z : M), Eq (HMul.hMul y (HMul.hMul x z)) (HMul. …
    -/
  · apply hc.imp
    /-
      case H₂
      M : Type u_4
      inst✝ : Monoid M
      l₁ l₂ : List M
      h : l₁.Perm l₂
      hc : List.Pairwise Commute l₁
      ⊢ ∀ {a b : M}, Commute a b → ∀ (z : M), Eq (HMul.hMul b (HMul.hMul a z)) (HMul …
    -/
    intro a b h z
    /-
      case H₂
      M : Type u_4
      inst✝ : Monoid M
      l₁ l₂ : List M
      h✝ : l₁.Perm l₂
      hc : List.Pairwise Commute l₁
      a b : M
      h : Commute a b
      z : M
      ⊢ Eq (HMul.hMul b (HMul.hMul a z)) (HMul.hMul a (HMul.hMul b z))
    -/
    rw [← mul_assoc, ← mul_assoc, h]
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp)]
lemma prod_erase [DecidableEq M] (ha : a ∈ l) : a * (l.erase a).prod = l.prod :=
  prod_erase_of_comm ha fun x _ y _ ↦ mul_comm x y


@[to_additive (attr := simp)]
lemma prod_map_erase [DecidableEq α] (f : α → M) {a} :
    ∀ {l : List α}, a ∈ l → f a * ((l.erase a).map f).prod = (l.map f).prod
  | b :: l, h => by
    /-
      α : Type u_2
      M : Type u_4
      inst✝¹ : CommMonoid M
      inst✝ : DecidableEq α
      f : α → M
      a b : α
      l : List α
      h : Membership.mem (List.cons b l) a
      ⊢ Eq (HMul.hMul (f a) (List.map f ((List.cons b l).erase a)).prod) (List.map f …
    -/
    obtain rfl | ⟨ne, h⟩ := List.eq_or_ne_mem_of_mem h
      /-
        case inl
        α : Type u_2
        M : Type u_4
        inst✝¹ : CommMonoid M
        inst✝ : DecidableEq α
        f : α → M
        a : α
        l : List α
        h : Membership.mem (List.cons a l) a
        ⊢ Eq (HMul.hMul (f a) (List.map f ((List.cons a l).erase a)).prod) (List.map f …
      -/
    · simp only [map, erase_cons_head, prod_cons]
      /-
        🎉 no goals
      -/
    · simp only [map, erase_cons_tail (not_beq_of_ne ne.symm), prod_cons, prod_map_erase _ h,
        mul_left_comm (f a) (f b)]


@[to_additive] lemma Perm.prod_eq (h : Perm l₁ l₂) : prod l₁ = prod l₂ := h.foldr_op_eq


@[to_additive] lemma prod_reverse (l : List M) : prod l.reverse = prod l := (reverse_perm l).prod_eq


@[to_additive]
lemma prod_mul_prod_eq_prod_zipWith_mul_prod_drop :
    ∀ l l' : List M,
      l.prod * l'.prod =
        (zipWith (· * ·) l l').prod * (l.drop l'.length).prod * (l'.drop l.length).prod
                 /-
                   M : Type u_4
                   inst✝ : CommMonoid M
                   ys : List M
                   ⊢ Eq (HMul.hMul List.nil.prod ys.prod) (HMul.hMul (HMul.hMul (List.zipWith (fu …
                 -/
  | [], ys => by simp [Nat.zero_le]
                 /-
                   🎉 no goals
                 -/
                 /-
                   M : Type u_4
                   inst✝ : CommMonoid M
                   xs : List M
                   ⊢ Eq (HMul.hMul xs.prod List.nil.prod) (HMul.hMul (HMul.hMul (List.zipWith (fu …
                 -/
  | xs, [] => by simp [Nat.zero_le]
                 /-
                   🎉 no goals
                 -/
  | x :: xs, y :: ys => by
    /-
      M : Type u_4
      inst✝ : CommMonoid M
      x : M
      xs : List M
      y : M
      ys : List M
      ⊢ Eq (HMul.hMul (List.cons x xs).prod (List.cons y ys).prod) (HMul.hMul (HMul. …
    -/
    simp only [drop, length, zipWith_cons_cons, prod_cons]
    conv =>
      lhs; rw [mul_assoc]; right; rw [mul_comm, mul_assoc]; right
      rw [mul_comm, prod_mul_prod_eq_prod_zipWith_mul_prod_drop xs ys]
    /-
      M : Type u_4
      inst✝ : CommMonoid M
      x : M
      xs : List M
      y : M
      ys : List M
      ⊢ Eq (HMul.hMul x (HMul.hMul y (HMul.hMul (HMul.hMul (List.zipWith (fun x1 x2  …
    -/
    simp [mul_assoc]
    /-
      🎉 no goals
    -/


@[to_additive]
lemma prod_mul_prod_eq_prod_zipWith_of_length_eq (l l' : List M) (h : l.length = l'.length) :
    l.prod * l'.prod = (zipWith (· * ·) l l').prod := by
  /-
    M : Type u_4
    inst✝ : CommMonoid M
    l l' : List M
    h : Eq l.length l'.length
    ⊢ Eq (HMul.hMul l.prod l'.prod) (List.zipWith (fun x1 x2 => HMul.hMul x1 x2) l …
  -/
  apply (prod_mul_prod_eq_prod_zipWith_mul_prod_drop l l').trans
  /-
    M : Type u_4
    inst✝ : CommMonoid M
    l l' : List M
    h : Eq l.length l'.length
    ⊢ Eq (HMul.hMul (HMul.hMul (List.zipWith (fun x1 x2 => HMul.hMul x1 x2) l l'). …
  -/
  rw [← h, drop_length, h, drop_length, prod_nil, mul_one, mul_one]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma prod_map_ite (p : α → Prop) [DecidablePred p] (f g : α → M) (l : List α) :
    (l.map fun a => if p a then f a else g a).prod =
      ((l.filter p).map f).prod * ((l.filter fun a ↦ ¬p a).map g).prod := by
  induction l with
  | nil => simp
  | cons x xs ih =>
    simp only [map_cons, filter_cons, prod_cons, nodup_cons, ne_eq, mem_cons, count_cons] at ih ⊢
    rw [ih]
    clear ih
    by_cases hx : p x
    · simp only [hx, ↓reduceIte, decide_not, decide_true, map_cons, prod_cons, not_true_eq_false,
        decide_false, Bool.false_eq_true, mul_assoc]
    · simp only [hx, ↓reduceIte, decide_not, decide_false, Bool.false_eq_true, not_false_eq_true,
      decide_true, map_cons, prod_cons, mul_left_comm]


@[to_additive]
lemma prod_map_filter_mul_prod_map_filter_not (p : α → Prop) [DecidablePred p] (f : α → M)
    (l : List α) :
    ((l.filter p).map f).prod * ((l.filter fun x => ¬p x).map f).prod = (l.map f).prod := by
  /-
    α : Type u_2
    M : Type u_4
    inst✝¹ : CommMonoid M
    p : α → Prop
    inst✝ : DecidablePred p
    f : α → M
    l : List α
    ⊢ Eq (HMul.hMul (List.map f (List.filter (fun b => Decidable.decide (p b)) l)) …
  -/
  rw [← prod_map_ite]
  /-
    α : Type u_2
    M : Type u_4
    inst✝¹ : CommMonoid M
    p : α → Prop
    inst✝ : DecidablePred p
    f : α → M
    l : List α
    ⊢ Eq (List.map (fun a => ite (p a) (f a) (f a)) l).prod (List.map f l).prod
  -/
  simp only [ite_self]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma eq_of_prod_take_eq [LeftCancelMonoid M] {L L' : List M} (h : L.length = L'.length)
    (h' : ∀ i ≤ L.length, (L.take i).prod = (L'.take i).prod) : L = L' := by
  /-
    M : Type u_4
    inst✝ : LeftCancelMonoid M
    L L' : List M
    h : Eq L.length L'.length
    h' : ∀ (i : Nat), LE.le i L.length → Eq (List.take i L).prod (List.take i L'). …
    ⊢ Eq L L'
  -/
  refine ext_get h fun i h₁ h₂ => ?_
  /-
    M : Type u_4
    inst✝ : LeftCancelMonoid M
    L L' : List M
    h : Eq L.length L'.length
    h' : ∀ (i : Nat), LE.le i L.length → Eq (List.take i L).prod (List.take i L'). …
    i : Nat
    h₁ : LT.lt i L.length
    h₂ : LT.lt i L'.length
    ⊢ Eq (L.get ⟨i, h₁⟩) (L'.get ⟨i, h₂⟩)
  -/
  have : (L.take (i + 1)).prod = (L'.take (i + 1)).prod := h' _ (Nat.succ_le_of_lt h₁)
  /-
    M : Type u_4
    inst✝ : LeftCancelMonoid M
    L L' : List M
    h : Eq L.length L'.length
    h' : ∀ (i : Nat), LE.le i L.length → Eq (List.take i L).prod (List.take i L'). …
    i : Nat
    h₁ : LT.lt i L.length
    h₂ : LT.lt i L'.length
    this : Eq (List.take (HAdd.hAdd i 1) L).prod (List.take (HAdd.hAdd i 1) L').prod
    ⊢ Eq (L.get ⟨i, h₁⟩) (L'.get ⟨i, h₂⟩)
  -/
  rw [prod_take_succ L i h₁, prod_take_succ L' i h₂, h' i (le_of_lt h₁)] at this
  /-
    M : Type u_4
    inst✝ : LeftCancelMonoid M
    L L' : List M
    h : Eq L.length L'.length
    h' : ∀ (i : Nat), LE.le i L.length → Eq (List.take i L).prod (List.take i L'). …
    i : Nat
    h₁ : LT.lt i L.length
    h₂ : LT.lt i L'.length
    this : Eq (HMul.hMul (List.take i L').prod (GetElem.getElem L i h₁)) (HMul.hMu …
    ⊢ Eq (L.get ⟨i, h₁⟩) (L'.get ⟨i, h₂⟩)
  -/
  convert mul_left_cancel this
  /-
    🎉 no goals
  -/


/-- This is the `List.prod` version of `mul_inv_rev` -/
@[to_additive "This is the `List.sum` version of `add_neg_rev`"]
theorem prod_inv_reverse : ∀ L : List G, L.prod⁻¹ = (L.map fun x => x⁻¹).reverse.prod
             /-
               G : Type u_7
               inst✝ : Group G
               ⊢ Eq (Inv.inv List.nil.prod) (List.map (fun x => Inv.inv x) List.nil).reverse. …
             -/
  | [] => by simp
             /-
               🎉 no goals
             -/
                  /-
                    G : Type u_7
                    inst✝ : Group G
                    x : G
                    xs : List G
                    ⊢ Eq (Inv.inv (List.cons x xs).prod) (List.map (fun x => Inv.inv x) (List.cons …
                  -/
  | x :: xs => by simp [prod_inv_reverse xs]
                  /-
                    🎉 no goals
                  -/


/-- A non-commutative variant of `List.prod_reverse` -/
@[to_additive "A non-commutative variant of `List.sum_reverse`"]
theorem prod_reverse_noncomm : ∀ L : List G, L.reverse.prod = (L.map fun x => x⁻¹).prod⁻¹ := by
  /-
    G : Type u_7
    inst✝ : Group G
    ⊢ ∀ (L : List G), Eq L.reverse.prod (Inv.inv (List.map (fun x => Inv.inv x) L) …
  -/
  simp [prod_inv_reverse]
  /-
    🎉 no goals
  -/


/-- Counterpart to `List.prod_take_succ` when we have an inverse operation -/
@[to_additive (attr := simp)
  "Counterpart to `List.sum_take_succ` when we have a negation operation"]
theorem prod_drop_succ :
    ∀ (L : List G) (i : ℕ) (p : i < L.length), (L.drop (i + 1)).prod = L[i]⁻¹ * (L.drop i).prod
  | [], _, p => False.elim (Nat.not_lt_zero _ p)
                       /-
                         G : Type u_7
                         inst✝ : Group G
                         head✝ : G
                         tail✝ : List G
                         x✝ : LT.lt 0 (List.cons head✝ tail✝).length
                         ⊢ Eq (List.drop (HAdd.hAdd 0 1) (List.cons head✝ tail✝)).prod (HMul.hMul (Inv. …
                       -/
  | _ :: _, 0, _ => by simp
                       /-
                         🎉 no goals
                       -/
  | _ :: xs, i + 1, p => prod_drop_succ xs i (Nat.lt_of_succ_lt_succ p)


/-- Cancellation of a telescoping product. -/
@[to_additive "Cancellation of a telescoping sum."]
theorem prod_range_div' (n : ℕ) (f : ℕ → G) :
    ((range n).map fun k ↦ f k / f (k + 1)).prod = f 0 / f n := by
  induction n with
  | zero => exact (div_self' (f 0)).symm
  | succ n h =>
    rw [range_succ, map_append, map_singleton, prod_append, prod_singleton, h, div_mul_div_cancel]


lemma prod_rotate_eq_one_of_prod_eq_one :
    ∀ {l : List G} (_ : l.prod = 1) (n : ℕ), (l.rotate n).prod = 1
                   /-
                     G : Type u_7
                     inst✝ : Group G
                     x✝¹ : Eq List.nil.prod 1
                     x✝ : Nat
                     ⊢ Eq (List.nil.rotate x✝).prod 1
                   -/
  | [], _, _ => by simp
                   /-
                     🎉 no goals
                   -/
  | a :: l, hl, n => by
    /-
      G : Type u_7
      inst✝ : Group G
      a : G
      l : List G
      hl : Eq (List.cons a l).prod 1
      n : Nat
      ⊢ Eq ((List.cons a l).rotate n).prod 1
    -/
    have : n % List.length (a :: l) ≤ List.length (a :: l) := le_of_lt (Nat.mod_lt _ (by simp))
    /-
      G : Type u_7
      inst✝ : Group G
      a : G
      l : List G
      hl : Eq (List.cons a l).prod 1
      n : Nat
      this : LE.le (HMod.hMod n (List.cons a l).length) (List.cons a l).length
      ⊢ Eq ((List.cons a l).rotate n).prod 1
    -/
    rw [← List.take_append_drop (n % List.length (a :: l)) (a :: l)] at hl
    rw [← rotate_mod, rotate_eq_drop_append_take this, List.prod_append, mul_eq_one_iff_inv_eq,
      ← one_mul (List.prod _)⁻¹, ← hl, List.prod_append, mul_assoc, mul_inv_cancel, mul_one]


/-- This is the `List.prod` version of `mul_inv` -/
@[to_additive "This is the `List.sum` version of `add_neg`"]
theorem prod_inv : ∀ L : List G, L.prod⁻¹ = (L.map fun x => x⁻¹).prod
             /-
               G : Type u_7
               inst✝ : CommGroup G
               ⊢ Eq (Inv.inv List.nil.prod) (List.map (fun x => Inv.inv x) List.nil).prod
             -/
  | [] => by simp
             /-
               🎉 no goals
             -/
                  /-
                    G : Type u_7
                    inst✝ : CommGroup G
                    x : G
                    xs : List G
                    ⊢ Eq (Inv.inv (List.cons x xs).prod) (List.map (fun x => Inv.inv x) (List.cons …
                  -/
  | x :: xs => by simp [mul_comm, prod_inv xs]
                  /-
                    🎉 no goals
                  -/


/-- Cancellation of a telescoping product. -/
@[to_additive "Cancellation of a telescoping sum."]
theorem prod_range_div (n : ℕ) (f : ℕ → G) :
    ((range n).map fun k ↦ f (k + 1) / f k).prod = f n / f 0 := by
  /-
    G : Type u_7
    inst✝ : CommGroup G
    n : Nat
    f : Nat → G
    ⊢ Eq (List.map (fun k => HDiv.hDiv (f (HAdd.hAdd k 1)) (f k)) (List.range n)). …
  -/
  have h : ((·⁻¹) ∘ fun k ↦ f (k + 1) / f k) = fun k ↦ f k / f (k + 1) := by ext; apply inv_div
  /-
    G : Type u_7
    inst✝ : CommGroup G
    n : Nat
    f : Nat → G
    h : Eq (Function.comp (fun x => Inv.inv x) fun k => HDiv.hDiv (f (HAdd.hAdd k  …
    ⊢ Eq (List.map (fun k => HDiv.hDiv (f (HAdd.hAdd k 1)) (f k)) (List.range n)). …
  -/
  rw [← inv_inj, prod_inv, map_map, inv_div, h, prod_range_div']
  /-
    🎉 no goals
  -/


/-- Alternative version of `List.prod_set` when the list is over a group -/
@[to_additive "Alternative version of `List.sum_set` when the list is over a group"]
theorem prod_set' (L : List G) (n : ℕ) (a : G) :
    (L.set n a).prod = L.prod * if hn : n < L.length then L[n]⁻¹ * a else 1 := by
  /-
    G : Type u_7
    inst✝ : CommGroup G
    L : List G
    n : Nat
    a : G
    ⊢ Eq (L.set n a).prod (HMul.hMul L.prod (dite (LT.lt n L.length) (fun hn => HM …
  -/
  refine (prod_set L n a).trans ?_
  /-
    G : Type u_7
    inst✝ : CommGroup G
    L : List G
    n : Nat
    a : G
    ⊢ Eq (HMul.hMul (HMul.hMul (List.take n L).prod (ite (LT.lt n L.length) a 1))  …
  -/
  split_ifs with hn
  · rw [mul_comm _ a, mul_assoc a, prod_drop_succ L n hn, mul_comm _ (drop n L).prod, ←
      mul_assoc (take n L).prod, prod_take_mul_prod_drop, mul_comm a, mul_assoc]
  · simp only [take_of_length_le (le_of_not_lt hn), prod_nil, mul_one,
      drop_eq_nil_of_le ((le_of_not_lt hn).trans n.le_succ)]


@[to_additive]
lemma prod_map_ite_eq {A : Type*} [DecidableEq A] (l : List A) (f g : A → G) (a : A) :
    (l.map fun x => if x = a then f x else g x).prod
      = (f a / g a) ^ (l.count a) * (l.map g).prod := by
  induction l with
  | nil => simp
  | cons x xs ih =>
    simp only [map_cons, prod_cons, nodup_cons, ne_eq, mem_cons, count_cons] at ih ⊢
    rw [ih]
    clear ih
    by_cases hx : x = a
    · simp only [hx, ite_true, div_pow, pow_add, pow_one, div_eq_mul_inv, mul_assoc, mul_comm,
        mul_left_comm, mul_inv_cancel_left, beq_self_eq_true]
    · simp only [hx, ite_false, ne_comm.mp hx, add_zero, mul_assoc, mul_comm (g x) _, beq_iff_eq]


theorem sum_const_nat (m n : ℕ) : sum (replicate m n) = m * n :=
  sum_replicate m n


/-- This relies on `default ℕ = 0`. -/
theorem headI_add_tail_sum (L : List ℕ) : L.headI + L.tail.sum = L.sum := by
  /-
    L : List Nat
    ⊢ Eq (HAdd.hAdd L.headI L.tail.sum) L.sum
  -/
              /-
                🎉 no goals
              -/
  cases L <;> simp
              /-
                🎉 no goals
              -/


/-- This relies on `default ℕ = 0`. -/
theorem headI_le_sum (L : List ℕ) : L.headI ≤ L.sum :=
  Nat.le.intro (headI_add_tail_sum L)


/-- This relies on `default ℕ = 0`. -/
theorem tail_sum (L : List ℕ) : L.tail.sum = L.sum - L.headI := by
  /-
    L : List Nat
    ⊢ Eq L.tail.sum (HSub.hSub L.sum L.headI)
  -/
  rw [← headI_add_tail_sum L, add_comm, Nat.add_sub_cancel_right]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem alternatingProd_nil : alternatingProd ([] : List α) = 1 :=
  rfl


@[to_additive (attr := simp)]
theorem alternatingProd_singleton (a : α) : alternatingProd [a] = a :=
  rfl


@[to_additive]
theorem alternatingProd_cons_cons' (a b : α) (l : List α) :
    alternatingProd (a :: b :: l) = a * b⁻¹ * alternatingProd l :=
  rfl


@[to_additive]
theorem alternatingProd_cons_cons [DivInvMonoid α] (a b : α) (l : List α) :
    alternatingProd (a :: b :: l) = a / b * alternatingProd l := by
  /-
    α : Type u_2
    inst✝ : DivInvMonoid α
    a b : α
    l : List α
    ⊢ Eq (List.cons a (List.cons b l)).alternatingProd (HMul.hMul (HDiv.hDiv a b)  …
  -/
  rw [div_eq_mul_inv, alternatingProd_cons_cons']
  /-
    🎉 no goals
  -/


@[to_additive]
theorem alternatingProd_cons' :
    ∀ (a : α) (l : List α), alternatingProd (a :: l) = a * (alternatingProd l)⁻¹
                /-
                  α : Type u_2
                  inst✝ : CommGroup α
                  a : α
                  ⊢ Eq (List.cons a List.nil).alternatingProd (HMul.hMul a (Inv.inv List.nil.alt …
                -/
  | a, [] => by rw [alternatingProd_nil, inv_one, mul_one, alternatingProd_singleton]
                /-
                  🎉 no goals
                -/
  | a, b :: l => by
    /-
      α : Type u_2
      inst✝ : CommGroup α
      a b : α
      l : List α
      ⊢ Eq (List.cons a (List.cons b l)).alternatingProd (HMul.hMul a (Inv.inv (List …
    -/
    rw [alternatingProd_cons_cons', alternatingProd_cons' b l, mul_inv, inv_inv, mul_assoc]
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp)]
theorem alternatingProd_cons (a : α) (l : List α) :
    alternatingProd (a :: l) = a / alternatingProd l := by
  /-
    α : Type u_2
    inst✝ : CommGroup α
    a : α
    l : List α
    ⊢ Eq (List.cons a l).alternatingProd (HDiv.hDiv a l.alternatingProd)
  -/
  rw [div_eq_mul_inv, alternatingProd_cons']
  /-
    🎉 no goals
  -/


lemma sum_nat_mod (l : List ℕ) (n : ℕ) : l.sum % n = (l.map (· % n)).sum % n := by
  induction l with
  | nil => simp only [Nat.zero_mod, map_nil]
  | cons a l ih =>
    simpa only [map_cons, sum_cons, Nat.mod_add_mod, Nat.add_mod_mod] using congr((a + $ih) % n)


lemma prod_nat_mod (l : List ℕ) (n : ℕ) : l.prod % n = (l.map (· % n)).prod % n := by
  induction l with
  | nil => simp only [Nat.zero_mod, map_nil]
  | cons a l ih =>
    simpa only [prod_cons, map_cons, Nat.mod_mul_mod, Nat.mul_mod_mod] using congr((a * $ih) % n)


lemma sum_int_mod (l : List ℤ) (n : ℤ) : l.sum % n = (l.map (· % n)).sum % n := by
  /-
    l : List Int
    n : Int
    ⊢ Eq (HMod.hMod l.sum n) (HMod.hMod (List.map (fun x => HMod.hMod x n) l).sum n)
  -/
                  /-
                    🎉 no goals
                  -/
  induction l <;> simp [Int.add_emod, *]
                  /-
                    🎉 no goals
                  -/


lemma prod_int_mod (l : List ℤ) (n : ℤ) : l.prod % n = (l.map (· % n)).prod % n := by
  /-
    l : List Int
    n : Int
    ⊢ Eq (HMod.hMod l.prod n) (HMod.hMod (List.map (fun x => HMod.hMod x n) l).pro …
  -/
                  /-
                    🎉 no goals
                  -/
  induction l <;> simp [Int.mul_emod, *]
                  /-
                    🎉 no goals
                  -/


/-- Summing the count of `x` over a list filtered by some `p` is just `countP` applied to `p` -/
theorem sum_map_count_dedup_filter_eq_countP (p : α → Bool) (l : List α) :
    ((l.dedup.filter p).map fun x => l.count x).sum = l.countP p := by
  induction l with
  | nil => simp
  | cons a as h =>
    simp_rw [List.countP_cons, List.count_cons, List.sum_map_add]
    congr 1
    · refine _root_.trans ?_ h
      by_cases ha : a ∈ as
      · simp [dedup_cons_of_mem ha]
      · simp only [dedup_cons_of_not_mem ha, List.filter]
        match p a with
        | true => simp only [List.map_cons, List.sum_cons, List.count_eq_zero.2 ha, zero_add]
        | false => simp only
    · simp only [beq_iff_eq]
      by_cases hp : p a
      · refine _root_.trans (sum_map_eq_nsmul_single a _ fun _ h _ => by simp [h.symm]) ?_
        simp [hp, count_dedup]
      · refine _root_.trans (List.sum_eq_zero fun n hn => ?_) (by simp [hp])
        obtain ⟨a', ha'⟩ := List.mem_map.1 hn
        split_ifs at ha' with ha
        · simp only [ha.symm, mem_filter, mem_dedup, find?, mem_cons, true_or, hp,
            and_false, false_and, reduceCtorEq] at ha'
        · exact ha'.2.symm


theorem sum_map_count_dedup_eq_length (l : List α) :
    (l.dedup.map fun x => l.count x).sum = l.length := by
  /-
    α : Type u_2
    inst✝ : DecidableEq α
    l : List α
    ⊢ Eq (List.map (fun x => List.count x l) l.dedup).sum l.length
  -/
  simpa using sum_map_count_dedup_filter_eq_countP (fun _ => True) l
  /-
    🎉 no goals
  -/


@[to_additive]
theorem map_list_prod {F : Type*} [FunLike F M N] [MonoidHomClass F M N] (f : F) (l : List M) :
    f l.prod = (l.map f).prod :=
  (l.prod_hom f).symm


@[to_additive]
protected theorem map_list_prod (f : M →* N) (l : List M) : f l.prod = (l.map f).prod :=
  map_list_prod f l


set_option linter.deprecated false in
@[simp, deprecated "No deprecation message was provided." (since := "2024-10-17")]
lemma Nat.sum_eq_listSum (l : List ℕ) : Nat.sum l = l.sum := rfl


lemma length_sigma {σ : α → Type*} (l₁ : List α) (l₂ : ∀ a, List (σ a)) :
    length (l₁.sigma l₂) = (l₁.map fun a ↦ length (l₂ a)).sum := by
  /-
    α : Type u_2
    σ : α → Type u_8
    l₁ : List α
    l₂ : (a : α) → List (σ a)
    ⊢ Eq (l₁.sigma l₂).length (List.map (fun a => (l₂ a).length) l₁).sum
  -/
  induction' l₁ with x l₁ IH
    /-
      case nil
      α : Type u_2
      σ : α → Type u_8
      l₂ : (a : α) → List (σ a)
      ⊢ Eq (List.nil.sigma l₂).length (List.map (fun a => (l₂ a).length) List.nil).sum
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_2
      σ : α → Type u_8
      l₂ : (a : α) → List (σ a)
      x : α
      l₁ : List α
      IH : Eq (l₁.sigma l₂).length (List.map (fun a => (l₂ a).length) l₁).sum
      ⊢ Eq ((List.cons x l₁).sigma l₂).length (List.map (fun a => (l₂ a).length) (Li …
    -/
  · simp only [sigma_cons, length_append, length_map, IH, map, sum_cons]
    /-
      🎉 no goals
    -/


lemma ranges_flatten : ∀ (l : List ℕ), l.ranges.flatten = range l.sum
  | [] => rfl
                 /-
                   a : Nat
                   l : List Nat
                   ⊢ Eq (List.cons a l).ranges.flatten (List.range (List.cons a l).sum)
                 -/
  | a :: l => by simp [ranges, ← map_flatten, ranges_flatten, range_add]
                 /-
                   🎉 no goals
                 -/


/-- The members of `l.ranges` have no duplicate -/
theorem ranges_nodup {l s : List ℕ} (hs : s ∈ ranges l) : s.Nodup :=
                                  /-
                                    l s : List Nat
                                    hs : Membership.mem l.ranges s
                                    ⊢ List.Pairwise (fun x1 x2 => Ne x1 x2) l.ranges.flatten
                                  -/
  (List.pairwise_flatten.mp <| by rw [ranges_flatten]; exact nodup_range _).1 s hs
                                                       /-
                                                         🎉 no goals
                                                       -/


@[deprecated (since := "2024-10-15")] alias ranges_join := ranges_flatten


/-- Any entry of any member of `l.ranges` is strictly smaller than `l.sum`. -/
lemma mem_mem_ranges_iff_lt_sum (l : List ℕ) {n : ℕ} :
    (∃ s ∈ l.ranges, n ∈ s) ↔ n < l.sum := by
  /-
    l : List Nat
    n : Nat
    ⊢ Iff (Exists fun s => And (Membership.mem l.ranges s) (Membership.mem s n)) ( …
  -/
  rw [← mem_range, ← ranges_flatten, mem_flatten]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-16")] alias length_bind := length_flatMap


@[deprecated (since := "2024-10-16")] alias countP_bind := countP_flatMap


@[deprecated (since := "2024-10-16")] alias count_bind := count_flatMap


/-- In a flatten, taking the first elements up to an index which is the sum of the lengths of the
first `i` sublists, is the same as taking the flatten of the first `i` sublists. -/
lemma take_sum_flatten (L : List (List α)) (i : ℕ) :
    L.flatten.take ((L.map length).take i).sum = (L.take i).flatten := by
  /-
    α : Type u_2
    L : List (List α)
    i : Nat
    ⊢ Eq (List.take (List.take i (List.map List.length L)).sum L.flatten) (List.ta …
  -/
  induction L generalizing i
    /-
      case nil
      α : Type u_2
      i : Nat
      ⊢ Eq (List.take (List.take i (List.map List.length List.nil)).sum List.nil.fla …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_2
      head✝ : List α
      tail✝ : List (List α)
      tail_ih✝ : ∀ (i : Nat), Eq (List.take (List.take i (List.map List.length tail✝ …
      i : Nat
      ⊢ Eq (List.take (List.take i (List.map List.length (List.cons head✝ tail✝))).s …
    -/
                /-
                  🎉 no goals
                -/
  · cases i <;> simp [take_append, *]
                /-
                  🎉 no goals
                -/


@[deprecated (since := "2024-10-15")] alias take_sum_join := take_sum_flatten


/-- In a flatten, dropping all the elements up to an index which is the sum of the lengths of the
first `i` sublists, is the same as taking the join after dropping the first `i` sublists. -/
lemma drop_sum_flatten (L : List (List α)) (i : ℕ) :
    L.flatten.drop ((L.map length).take i).sum = (L.drop i).flatten := by
  /-
    α : Type u_2
    L : List (List α)
    i : Nat
    ⊢ Eq (List.drop (List.take i (List.map List.length L)).sum L.flatten) (List.dr …
  -/
  induction L generalizing i
    /-
      case nil
      α : Type u_2
      i : Nat
      ⊢ Eq (List.drop (List.take i (List.map List.length List.nil)).sum List.nil.fla …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_2
      head✝ : List α
      tail✝ : List (List α)
      tail_ih✝ : ∀ (i : Nat), Eq (List.drop (List.take i (List.map List.length tail✝ …
      i : Nat
      ⊢ Eq (List.drop (List.take i (List.map List.length (List.cons head✝ tail✝))).s …
    -/
                /-
                  🎉 no goals
                -/
  · cases i <;> simp [take_append, *]
                /-
                  🎉 no goals
                -/


@[deprecated (since := "2024-10-15")] alias drop_sum_join := drop_sum_flatten


/-- In a flatten of sublists, taking the slice between the indices `A` and `B - 1` gives back the
original sublist of index `i` if `A` is the sum of the lengths of sublists of index `< i`, and
`B` is the sum of the lengths of sublists of index `≤ i`. -/
lemma drop_take_succ_flatten_eq_getElem (L : List (List α)) (i : Nat) (h : i < L.length) :
    (L.flatten.take ((L.map length).take (i + 1)).sum).drop ((L.map length).take i).sum = L[i] := by
  have : (L.map length).take i = ((L.take (i + 1)).map length).take i := by
    simp [map_take, take_take, Nat.min_eq_left]
  simp only [this, length_map, take_sum_flatten, drop_sum_flatten,
    drop_take_succ_eq_cons_getElem, h, flatten, append_nil]


@[deprecated (since := "2024-06-11")]
alias drop_take_succ_join_eq_getElem := drop_take_succ_flatten_eq_getElem


@[deprecated drop_take_succ_flatten_eq_getElem (since := "2024-06-11")]
lemma drop_take_succ_join_eq_get (L : List (List α)) (i : Fin L.length) :
    (L.flatten.take ((L.map length).take (i + 1)).sum).drop
      ((L.map length).take i).sum = get L i := by
  /-
    α : Type u_2
    L : List (List α)
    i : Fin L.length
    ⊢ Eq (List.drop (List.take (↑i) (List.map List.length L)).sum (List.take (List …
  -/
  rw [drop_take_succ_flatten_eq_getElem _ _ i.2]
  /-
    α : Type u_2
    L : List (List α)
    i : Fin L.length
    ⊢ Eq (GetElem.getElem L ↑i ⋯) (L.get i)
  -/
  simp
  /-
    🎉 no goals
  -/


/-- If a product of integers is `-1`, then at least one factor must be `-1`. -/
theorem neg_one_mem_of_prod_eq_neg_one {l : List ℤ} (h : l.prod = -1) : (-1 : ℤ) ∈ l := by
  /-
    l : List Int
    h : Eq l.prod (-1)
    ⊢ Membership.mem l (-1)
  -/
  obtain ⟨x, h₁, h₂⟩ := exists_mem_ne_one_of_prod_ne_one (ne_of_eq_of_ne h (by decide))
  exact Or.resolve_left
    (Int.isUnit_iff.mp (prod_isUnit_iff.mp
      (h.symm ▸ ⟨⟨-1, -1, by decide, by decide⟩, rfl⟩ : IsUnit l.prod) x h₁)) h₂ ▸ h₁


/-- If all elements in a list are bounded below by `1`, then the length of the list is bounded
by the sum of the elements. -/
theorem length_le_sum_of_one_le (L : List ℕ) (h : ∀ i ∈ L, 1 ≤ i) : L.length ≤ L.sum := by
  induction L with
  | nil => simp
  | cons j L IH =>
    rw [sum_cons, length, add_comm]
    exact Nat.add_le_add (h _ (mem_cons_self _ _)) (IH fun i hi => h i (mem_cons.2 (Or.inr hi)))


theorem dvd_prod [CommMonoid M] {a} {l : List M} (ha : a ∈ l) : a ∣ l.prod := by
  /-
    M : Type u_4
    inst✝ : CommMonoid M
    a : M
    l : List M
    ha : Membership.mem l a
    ⊢ Dvd.dvd a l.prod
  -/
  let ⟨s, t, h⟩ := append_of_mem ha
  /-
    M : Type u_4
    inst✝ : CommMonoid M
    a : M
    l : List M
    ha : Membership.mem l a
    s t : List M
    h : Eq l (HAppend.hAppend s (List.cons a t))
    ⊢ Dvd.dvd a l.prod
  -/
  rw [h, prod_append, prod_cons, mul_left_comm]
  /-
    M : Type u_4
    inst✝ : CommMonoid M
    a : M
    l : List M
    ha : Membership.mem l a
    s t : List M
    h : Eq l (HAppend.hAppend s (List.cons a t))
    ⊢ Dvd.dvd a (HMul.hMul a (HMul.hMul s.prod t.prod))
  -/
  exact dvd_mul_right _ _
  /-
    🎉 no goals
  -/


theorem Sublist.prod_dvd_prod [CommMonoid M] {l₁ l₂ : List M} (h : l₁ <+ l₂) :
    l₁.prod ∣ l₂.prod := by
  /-
    M : Type u_4
    inst✝ : CommMonoid M
    l₁ l₂ : List M
    h : l₁.Sublist l₂
    ⊢ Dvd.dvd l₁.prod l₂.prod
  -/
  obtain ⟨l, hl⟩ := h.exists_perm_append
  /-
    case intro
    M : Type u_4
    inst✝ : CommMonoid M
    l₁ l₂ : List M
    h : l₁.Sublist l₂
    l : List M
    hl : l₂.Perm (HAppend.hAppend l₁ l)
    ⊢ Dvd.dvd l₁.prod l₂.prod
  -/
  rw [hl.prod_eq, prod_append]
  /-
    case intro
    M : Type u_4
    inst✝ : CommMonoid M
    l₁ l₂ : List M
    h : l₁.Sublist l₂
    l : List M
    hl : l₂.Perm (HAppend.hAppend l₁ l)
    ⊢ Dvd.dvd l₁.prod (HMul.hMul l₁.prod l.prod)
  -/
  exact dvd_mul_right _ _
  /-
    🎉 no goals
  -/


@[to_additive]
theorem alternatingProd_append :
    ∀ l₁ l₂ : List α,
      alternatingProd (l₁ ++ l₂) = alternatingProd l₁ * alternatingProd l₂ ^ (-1 : ℤ) ^ length l₁
                 /-
                   α : Type u_2
                   inst✝ : CommGroup α
                   l₂ : List α
                   ⊢ Eq (HAppend.hAppend List.nil l₂).alternatingProd (HMul.hMul List.nil.alterna …
                 -/
  | [], l₂ => by simp
                 /-
                   🎉 no goals
                 -/
  | a :: l₁, l₂ => by
    simp_rw [cons_append, alternatingProd_cons, alternatingProd_append, length_cons, pow_succ',
      Int.neg_mul, one_mul, zpow_neg, ← div_eq_mul_inv, div_div]


@[to_additive]
theorem alternatingProd_reverse :
    ∀ l : List α, alternatingProd (reverse l) = alternatingProd l ^ (-1 : ℤ) ^ (length l + 1)
             /-
               α : Type u_2
               inst✝ : CommGroup α
               ⊢ Eq List.nil.reverse.alternatingProd (HPow.hPow List.nil.alternatingProd (HPo …
             -/
  | [] => by simp only [alternatingProd_nil, one_zpow, reverse_nil]
             /-
               🎉 no goals
             -/
  | a :: l => by
    simp_rw [reverse_cons, alternatingProd_append, alternatingProd_reverse,
      alternatingProd_singleton, alternatingProd_cons, length_reverse, length, pow_succ',
      Int.neg_mul, one_mul, zpow_neg, inv_inv]
    /-
      α : Type u_2
      inst✝ : CommGroup α
      a : α
      l : List α
      ⊢ Eq (HMul.hMul (Inv.inv (HPow.hPow l.alternatingProd (HPow.hPow (-1) l.length …
    -/
    rw [mul_comm, ← div_eq_mul_inv, div_zpow]
    /-
      🎉 no goals
    -/


lemma op_list_prod : ∀ l : List M, op l.prod = (l.map op).reverse.prod := by
  /-
    M : Type u_4
    inst✝ : Monoid M
    ⊢ ∀ (l : List M), Eq (MulOpposite.op l.prod) (List.map MulOpposite.op l).rever …
  -/
  intro l; induction l with
  | nil => rfl
  | cons x xs ih =>
    rw [List.prod_cons, List.map_cons, List.reverse_cons', List.prod_concat, op_mul, ih]


lemma unop_list_prod (l : List Mᵐᵒᵖ) : l.prod.unop = (l.map unop).reverse.prod := by
  rw [← op_inj, op_unop, MulOpposite.op_list_prod, map_reverse, map_map, reverse_reverse,
    op_comp_unop, map_id]


/-- A morphism into the opposite monoid acts on the product by acting on the reversed elements. -/
lemma unop_map_list_prod {F : Type*} [FunLike F M Nᵐᵒᵖ] [MonoidHomClass F M Nᵐᵒᵖ]
    (f : F) (l : List M) :
    (f l.prod).unop = (l.map (MulOpposite.unop ∘ f)).reverse.prod := by
  /-
    M : Type u_4
    N : Type u_5
    inst✝³ : Monoid M
    inst✝² : Monoid N
    F : Type u_8
    inst✝¹ : FunLike F M (MulOpposite N)
    inst✝ : MonoidHomClass F M (MulOpposite N)
    f : F
    l : List M
    ⊢ Eq (MulOpposite.unop (f l.prod)) (List.map (Function.comp MulOpposite.unop ⇑ …
  -/
  rw [map_list_prod f l, MulOpposite.unop_list_prod, List.map_map]
  /-
    🎉 no goals
  -/


