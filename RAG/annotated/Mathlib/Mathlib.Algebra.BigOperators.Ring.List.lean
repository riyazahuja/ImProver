lemma list_sum_right (a : R) (l : List R) (h : ∀ b ∈ l, Commute a b) : Commute a l.sum := by
  induction l with
  | nil => exact Commute.zero_right _
  | cons x xs ih =>
    rw [List.sum_cons]
    exact (h _ <| mem_cons_self _ _).add_right (ih fun j hj ↦ h _ <| mem_cons_of_mem _ hj)


lemma list_sum_left (b : R) (l : List R) (h : ∀ a ∈ l, Commute a b) : Commute l.sum b :=
  ((Commute.list_sum_right _ _) fun _x hx ↦ (h _ hx).symm).symm


@[simp]
lemma prod_map_neg (l : List M) :
    (l.map Neg.neg).prod = (-1) ^ l.length * l.prod := by
  /-
    M : Type u_3
    inst✝¹ : CommMonoid M
    inst✝ : HasDistribNeg M
    l : List M
    ⊢ Eq (List.map Neg.neg l).prod (HMul.hMul (HPow.hPow (-1) l.length) l.prod)
  -/
                  /-
                    🎉 no goals
                  -/
  induction l <;> simp [*, pow_succ, ((Commute.neg_one_left _).pow_left _).left_comm]
                  /-
                    🎉 no goals
                  -/


/-- If zero is an element of a list `l`, then `List.prod l = 0`. If the domain is a nontrivial
monoid with zero with no divisors, then this implication becomes an `iff`, see
`List.prod_eq_zero_iff`. -/
lemma prod_eq_zero : ∀ {l : List M₀}, (0 : M₀) ∈ l → l.prod = 0
  -- |  absurd h (not_mem_nil _)
  | a :: l, h => by
    /-
      M₀ : Type u_4
      inst✝ : MonoidWithZero M₀
      a : M₀
      l : List M₀
      h : Membership.mem (List.cons a l) 0
      ⊢ Eq (List.cons a l).prod 0
    -/
    rw [prod_cons]
    /-
      M₀ : Type u_4
      inst✝ : MonoidWithZero M₀
      a : M₀
      l : List M₀
      h : Membership.mem (List.cons a l) 0
      ⊢ Eq (HMul.hMul a l.prod) 0
    -/
    rcases mem_cons.1 h with ha | hl
    /-
      case inl
      M₀ : Type u_4
      inst✝ : MonoidWithZero M₀
      a : M₀
      l : List M₀
      h : Membership.mem (List.cons a l) 0
      ha : Eq 0 a
      ⊢ Eq (HMul.hMul a l.prod) 0
    -/
    exacts [mul_eq_zero_of_left ha.symm _, mul_eq_zero_of_right _ (prod_eq_zero hl)]
    /-
      🎉 no goals
    -/


/-- Product of elements of a list `l` equals zero if and only if `0 ∈ l`. See also
`List.prod_eq_zero` for an implication that needs weaker typeclass assumptions. -/
@[simp] lemma prod_eq_zero_iff : ∀ {l : List M₀}, l.prod = 0 ↔ (0 : M₀) ∈ l
             /-
               M₀ : Type u_4
               inst✝² : MonoidWithZero M₀
               inst✝¹ : Nontrivial M₀
               inst✝ : NoZeroDivisors M₀
               ⊢ Iff (Eq List.nil.prod 0) (Membership.mem List.nil 0)
             -/
  | [] => by simp
             /-
               🎉 no goals
             -/
                 /-
                   M₀ : Type u_4
                   inst✝² : MonoidWithZero M₀
                   inst✝¹ : Nontrivial M₀
                   inst✝ : NoZeroDivisors M₀
                   a : M₀
                   l : List M₀
                   ⊢ Iff (Eq (List.cons a l).prod 0) (Membership.mem (List.cons a l) 0)
                 -/
  | a :: l => by rw [prod_cons, mul_eq_zero, prod_eq_zero_iff, mem_cons, eq_comm]
                 /-
                   🎉 no goals
                 -/


lemma prod_ne_zero (hL : (0 : M₀) ∉ l) : l.prod ≠ 0 := mt prod_eq_zero_iff.1 hL


lemma sum_map_mul_left : (l.map fun b ↦ r * f b).sum = r * (l.map f).sum :=
  sum_map_hom l f <| AddMonoidHom.mulLeft r


lemma sum_map_mul_right : (l.map fun b ↦ f b * r).sum = (l.map f).sum * r :=
  sum_map_hom l f <| AddMonoidHom.mulRight r


lemma dvd_sum [NonUnitalSemiring R] {a} {l : List R} (h : ∀ x ∈ l, a ∣ x) : a ∣ l.sum := by
  induction l with
  | nil => exact dvd_zero _
  | cons x l ih =>
    rw [List.sum_cons]
    exact dvd_add (h _ (mem_cons_self _ _)) (ih fun x hx ↦ h x (mem_cons_of_mem _ hx))


@[simp] lemma sum_zipWith_distrib_left [Semiring R] (f : ι → κ → R) (a : R) :
    ∀ (l₁ : List ι) (l₂ : List κ),
      (zipWith (fun i j ↦ a * f i j) l₁ l₂).sum = a * (zipWith f l₁ l₂).sum
                /-
                  ι : Type u_1
                  κ : Type u_2
                  R : Type u_5
                  inst✝ : Semiring R
                  f : ι → κ → R
                  a : R
                  x✝ : List κ
                  ⊢ Eq (List.zipWith (fun i j => HMul.hMul a (f i j)) List.nil x✝).sum (HMul.hMu …
                -/
  | [], _ => by simp
                /-
                  🎉 no goals
                -/
                /-
                  ι : Type u_1
                  κ : Type u_2
                  R : Type u_5
                  inst✝ : Semiring R
                  f : ι → κ → R
                  a : R
                  x✝ : List ι
                  ⊢ Eq (List.zipWith (fun i j => HMul.hMul a (f i j)) x✝ List.nil).sum (HMul.hMu …
                -/
  | _, [] => by simp
                /-
                  🎉 no goals
                -/
                           /-
                             ι : Type u_1
                             κ : Type u_2
                             R : Type u_5
                             inst✝ : Semiring R
                             f : ι → κ → R
                             a : R
                             i : ι
                             l₁ : List ι
                             j : κ
                             l₂ : List κ
                             ⊢ Eq (List.zipWith (fun i j => HMul.hMul a (f i j)) (List.cons i l₁) (List.con …
                           -/
  | i :: l₁, j :: l₂ => by simp [sum_zipWith_distrib_left, mul_add]
                           /-
                             🎉 no goals
                           -/


