                                                                                     /-
                                                                                       α : Type u
                                                                                       l : List α
                                                                                       n : Nat
                                                                                       ⊢ Eq (l.rotate (HMod.hMod n l.length)) (l.rotate n)
                                                                                     -/
theorem rotate_mod (l : List α) (n : ℕ) : l.rotate (n % l.length) = l.rotate n := by simp [rotate]
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


@[simp]
                                                               /-
                                                                 α : Type u
                                                                 n : Nat
                                                                 ⊢ Eq (List.nil.rotate n) List.nil
                                                               -/
theorem rotate_nil (n : ℕ) : ([] : List α).rotate n = [] := by simp [rotate]
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[simp]
                                                        /-
                                                          α : Type u
                                                          l : List α
                                                          ⊢ Eq (l.rotate 0) l
                                                        -/
theorem rotate_zero (l : List α) : l.rotate 0 = l := by simp [rotate]
                                                        /-
                                                          🎉 no goals
                                                        -/

-- Porting note: removing simp, simp can prove it

                                                                 /-
                                                                   α : Type u
                                                                   n : Nat
                                                                   ⊢ Eq (List.nil.rotate' n) List.nil
                                                                 -/
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
theorem rotate'_nil (n : ℕ) : ([] : List α).rotate' n = [] := by cases n <;> rfl
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


@[simp]
                                                          /-
                                                            α : Type u
                                                            l : List α
                                                            ⊢ Eq (l.rotate' 0) l
                                                          -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
theorem rotate'_zero (l : List α) : l.rotate' 0 = l := by cases l <;> rfl
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem rotate'_cons_succ (l : List α) (a : α) (n : ℕ) :
                                                                  /-
                                                                    α : Type u
                                                                    l : List α
                                                                    a : α
                                                                    n : Nat
                                                                    ⊢ Eq ((List.cons a l).rotate' n.succ) ((HAppend.hAppend l (List.cons a List.ni …
                                                                  -/
    (a :: l : List α).rotate' n.succ = (l ++ [a]).rotate' n := by simp [rotate']
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[simp]
theorem length_rotate' : ∀ (l : List α) (n : ℕ), (l.rotate' n).length = l.length
                /-
                  α : Type u
                  x✝ : Nat
                  ⊢ Eq (List.nil.rotate' x✝).length List.nil.length
                -/
  | [], _ => by simp
                /-
                  🎉 no goals
                -/
  | _ :: _, 0 => rfl
                        /-
                          α : Type u
                          a : α
                          l : List α
                          n : Nat
                          ⊢ Eq ((List.cons a l).rotate' (HAdd.hAdd n 1)).length (List.cons a l).length
                        -/
  | a :: l, n + 1 => by rw [List.rotate', length_rotate' (l ++ [a]) n]; simp
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


theorem rotate'_eq_drop_append_take :
    ∀ {l : List α} {n : ℕ}, n ≤ l.length → l.rotate' n = l.drop n ++ l.take n
                   /-
                     α : Type u
                     n : Nat
                     h : LE.le n List.nil.length
                     ⊢ Eq (List.nil.rotate' n) (HAppend.hAppend (List.drop n List.nil) (List.take n …
                   -/
  | [], n, h => by simp [drop_append_of_le_length h]
                   /-
                     🎉 no goals
                   -/
                  /-
                    α : Type u
                    l : List α
                    h : LE.le 0 l.length
                    ⊢ Eq (l.rotate' 0) (HAppend.hAppend (List.drop 0 l) (List.take 0 l))
                  -/
  | l, 0, h => by simp [take_append_of_le_length h]
                  /-
                    🎉 no goals
                  -/
  | a :: l, n + 1, h => by
    /-
      α : Type u
      a : α
      l : List α
      n : Nat
      h : LE.le (HAdd.hAdd n 1) (List.cons a l).length
      ⊢ Eq ((List.cons a l).rotate' (HAdd.hAdd n 1)) (HAppend.hAppend (List.drop (HA …
    -/
    have hnl : n ≤ l.length := le_of_succ_le_succ h
    have hnl' : n ≤ (l ++ [a]).length := by
      rw [length_append, length_cons, List.length]; exact le_of_succ_le h
    rw [rotate'_cons_succ, rotate'_eq_drop_append_take hnl', drop, take,
                                                                     /-
                                                                       α : Type u
                                                                       a : α
                                                                       l : List α
                                                                       n : Nat
                                                                       h : LE.le (HAdd.hAdd n 1) (List.cons a l).length
                                                                       hnl : LE.le n l.length
                                                                       hnl' : LE.le n (HAppend.hAppend l (List.cons a List.nil)).length
                                                                       ⊢ Eq (HAppend.hAppend (HAppend.hAppend (List.drop n l) (List.cons a List.nil)) …
                                                                     -/
        drop_append_of_le_length hnl, take_append_of_le_length hnl]; simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem rotate'_rotate' : ∀ (l : List α) (n m : ℕ), (l.rotate' n).rotate' m = l.rotate' (n + m)
                       /-
                         α : Type u
                         a : α
                         l : List α
                         m : Nat
                         ⊢ Eq (((List.cons a l).rotate' 0).rotate' m) ((List.cons a l).rotate' (HAdd.hA …
                       -/
  | a :: l, 0, m => by simp
                       /-
                         🎉 no goals
                       -/
                   /-
                     α : Type u
                     n m : Nat
                     ⊢ Eq ((List.nil.rotate' n).rotate' m) (List.nil.rotate' (HAdd.hAdd n m))
                   -/
  | [], n, m => by simp
                   /-
                     🎉 no goals
                   -/
  | a :: l, n + 1, m => by
    rw [rotate'_cons_succ, rotate'_rotate' _ n, Nat.add_right_comm, ← rotate'_cons_succ,
      Nat.succ_eq_add_one]


@[simp]
theorem rotate'_length (l : List α) : rotate' l l.length = l := by
  /-
    α : Type u
    l : List α
    ⊢ Eq (l.rotate' l.length) l
  -/
  rw [rotate'_eq_drop_append_take le_rfl]; simp
                                           /-
                                             🎉 no goals
                                           -/


@[simp]
theorem rotate'_length_mul (l : List α) : ∀ n : ℕ, l.rotate' (l.length * n) = l
            /-
              α : Type u
              l : List α
              ⊢ Eq (l.rotate' (HMul.hMul l.length 0)) l
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
  | n + 1 =>
    calc
      l.rotate' (l.length * (n + 1)) =
          (l.rotate' (l.length * n)).rotate' (l.rotate' (l.length * n)).length := by
        /-
          α : Type u
          l : List α
          n : Nat
          ⊢ Eq (l.rotate' (HMul.hMul l.length (HAdd.hAdd n 1))) ((l.rotate' (HMul.hMul l …
        -/
        simp [-rotate'_length, Nat.mul_succ, rotate'_rotate']
        /-
          🎉 no goals
        -/
                  /-
                    α : Type u
                    l : List α
                    n : Nat
                    ⊢ Eq ((l.rotate' (HMul.hMul l.length n)).rotate' (l.rotate' (HMul.hMul l.lengt …
                  -/
      _ = l := by rw [rotate'_length, rotate'_length_mul l n]
                  /-
                    🎉 no goals
                  -/


theorem rotate'_mod (l : List α) (n : ℕ) : l.rotate' (n % l.length) = l.rotate' n :=
  calc
    l.rotate' (n % l.length) =
        (l.rotate' (n % l.length)).rotate' ((l.rotate' (n % l.length)).length * (n / l.length)) :=
         /-
           α : Type u
           l : List α
           n : Nat
           ⊢ Eq (l.rotate' (HMod.hMod n l.length)) ((l.rotate' (HMod.hMod n l.length)).ro …
         -/
      by rw [rotate'_length_mul]
         /-
           🎉 no goals
         -/
                          /-
                            α : Type u
                            l : List α
                            n : Nat
                            ⊢ Eq ((l.rotate' (HMod.hMod n l.length)).rotate' (HMul.hMul (l.rotate' (HMod.h …
                          -/
    _ = l.rotate' n := by rw [rotate'_rotate', length_rotate', Nat.mod_add_div]
                          /-
                            🎉 no goals
                          -/


theorem rotate_eq_rotate' (l : List α) (n : ℕ) : l.rotate n = l.rotate' n :=
                              /-
                                α : Type u
                                l : List α
                                n : Nat
                                h : Eq l.length 0
                                ⊢ Eq (l.rotate n) (l.rotate' n)
                              -/
  if h : l.length = 0 then by simp_all [length_eq_zero]
                              /-
                                🎉 no goals
                              -/
  else by
    rw [← rotate'_mod,
        rotate'_eq_drop_append_take (le_of_lt (Nat.mod_lt _ (Nat.pos_of_ne_zero h)))]
    /-
      α : Type u
      l : List α
      n : Nat
      h : Not (Eq l.length 0)
      ⊢ Eq (l.rotate n) (HAppend.hAppend (List.drop (HMod.hMod n l.length) l) (List. …
    -/
    simp [rotate]
    /-
      🎉 no goals
    -/


theorem rotate_cons_succ (l : List α) (a : α) (n : ℕ) :
    (a :: l : List α).rotate (n + 1) = (l ++ [a]).rotate n := by
  /-
    α : Type u
    l : List α
    a : α
    n : Nat
    ⊢ Eq ((List.cons a l).rotate (HAdd.hAdd n 1)) ((HAppend.hAppend l (List.cons a …
  -/
  rw [rotate_eq_rotate', rotate_eq_rotate', rotate'_cons_succ]
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_rotate : ∀ {l : List α} {a : α} {n : ℕ}, a ∈ l.rotate n ↔ a ∈ l
                   /-
                     α : Type u
                     x✝ : α
                     n : Nat
                     ⊢ Iff (Membership.mem (List.nil.rotate n) x✝) (Membership.mem List.nil x✝)
                   -/
  | [], _, n => by simp
                   /-
                     🎉 no goals
                   -/
                       /-
                         α : Type u
                         a : α
                         l : List α
                         x✝ : α
                         ⊢ Iff (Membership.mem ((List.cons a l).rotate 0) x✝) (Membership.mem (List.con …
                       -/
  | a :: l, _, 0 => by simp
                       /-
                         🎉 no goals
                       -/
                           /-
                             α : Type u
                             a : α
                             l : List α
                             x✝ : α
                             n : Nat
                             ⊢ Iff (Membership.mem ((List.cons a l).rotate (HAdd.hAdd n 1)) x✝) (Membership …
                           -/
  | a :: l, _, n + 1 => by simp [rotate_cons_succ, mem_rotate, or_comm]
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem length_rotate (l : List α) (n : ℕ) : (l.rotate n).length = l.length := by
  /-
    α : Type u
    l : List α
    n : Nat
    ⊢ Eq (l.rotate n).length l.length
  -/
  rw [rotate_eq_rotate', length_rotate']
  /-
    🎉 no goals
  -/


@[simp]
theorem rotate_replicate (a : α) (n : ℕ) (k : ℕ) : (replicate n a).rotate k = replicate n a :=
                         /-
                           α : Type u
                           a : α
                           n k : Nat
                           ⊢ Eq ((List.replicate n a).rotate k).length n
                         -/
  eq_replicate_iff.2 ⟨by rw [length_rotate, length_replicate], fun b hb =>
                         /-
                           🎉 no goals
                         -/
    eq_of_mem_replicate <| mem_rotate.1 hb⟩


theorem rotate_eq_drop_append_take {l : List α} {n : ℕ} :
    n ≤ l.length → l.rotate n = l.drop n ++ l.take n := by
  /-
    α : Type u
    l : List α
    n : Nat
    ⊢ LE.le n l.length → Eq (l.rotate n) (HAppend.hAppend (List.drop n l) (List.ta …
  -/
  rw [rotate_eq_rotate']; exact rotate'_eq_drop_append_take
                          /-
                            🎉 no goals
                          -/


theorem rotate_eq_drop_append_take_mod {l : List α} {n : ℕ} :
    l.rotate n = l.drop (n % l.length) ++ l.take (n % l.length) := by
  /-
    α : Type u
    l : List α
    n : Nat
    ⊢ Eq (l.rotate n) (HAppend.hAppend (List.drop (HMod.hMod n l.length) l) (List. …
  -/
  rcases l.length.zero_le.eq_or_lt with hl | hl
    /-
      case inl
      α : Type u
      l : List α
      n : Nat
      hl : Eq 0 l.length
      ⊢ Eq (l.rotate n) (HAppend.hAppend (List.drop (HMod.hMod n l.length) l) (List. …
    -/
  · simp [eq_nil_of_length_eq_zero hl.symm]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u
    l : List α
    n : Nat
    hl : LT.lt 0 l.length
    ⊢ Eq (l.rotate n) (HAppend.hAppend (List.drop (HMod.hMod n l.length) l) (List. …
  -/
  rw [← rotate_eq_drop_append_take (n.mod_lt hl).le, rotate_mod]
  /-
    🎉 no goals
  -/


@[simp]
theorem rotate_append_length_eq (l l' : List α) : (l ++ l').rotate l.length = l' ++ l := by
  /-
    α : Type u
    l l' : List α
    ⊢ Eq ((HAppend.hAppend l l').rotate l.length) (HAppend.hAppend l' l)
  -/
  rw [rotate_eq_rotate']
  /-
    α : Type u
    l l' : List α
    ⊢ Eq ((HAppend.hAppend l l').rotate' l.length) (HAppend.hAppend l' l)
  -/
  induction l generalizing l'
    /-
      case nil
      α : Type u
      l' : List α
      ⊢ Eq ((HAppend.hAppend List.nil l').rotate' List.nil.length) (HAppend.hAppend  …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      head✝ : α
      tail✝ : List α
      tail_ih✝ : ∀ (l' : List α), Eq ((HAppend.hAppend tail✝ l').rotate' tail✝.lengt …
      l' : List α
      ⊢ Eq ((HAppend.hAppend (List.cons head✝ tail✝) l').rotate' (List.cons head✝ ta …
    -/
  · simp_all [rotate']
    /-
      🎉 no goals
    -/


theorem rotate_rotate (l : List α) (n m : ℕ) : (l.rotate n).rotate m = l.rotate (n + m) := by
  /-
    α : Type u
    l : List α
    n m : Nat
    ⊢ Eq ((l.rotate n).rotate m) (l.rotate (HAdd.hAdd n m))
  -/
  rw [rotate_eq_rotate', rotate_eq_rotate', rotate_eq_rotate', rotate'_rotate']
  /-
    🎉 no goals
  -/


@[simp]
theorem rotate_length (l : List α) : rotate l l.length = l := by
  /-
    α : Type u
    l : List α
    ⊢ Eq (l.rotate l.length) l
  -/
  rw [rotate_eq_rotate', rotate'_length]
  /-
    🎉 no goals
  -/


@[simp]
theorem rotate_length_mul (l : List α) (n : ℕ) : l.rotate (l.length * n) = l := by
  /-
    α : Type u
    l : List α
    n : Nat
    ⊢ Eq (l.rotate (HMul.hMul l.length n)) l
  -/
  rw [rotate_eq_rotate', rotate'_length_mul]
  /-
    🎉 no goals
  -/


theorem rotate_perm (l : List α) (n : ℕ) : l.rotate n ~ l := by
  /-
    α : Type u
    l : List α
    n : Nat
    ⊢ (l.rotate n).Perm l
  -/
  rw [rotate_eq_rotate']
  /-
    α : Type u
    l : List α
    n : Nat
    ⊢ (l.rotate' n).Perm l
  -/
  induction' n with n hn generalizing l
    /-
      case zero
      α : Type u
      l : List α
      ⊢ (l.rotate' 0).Perm l
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u
      n : Nat
      hn : ∀ (l : List α), (l.rotate' n).Perm l
      l : List α
      ⊢ (l.rotate' (HAdd.hAdd n 1)).Perm l
    -/
  · cases' l with hd tl
      /-
        case succ.nil
        α : Type u
        n : Nat
        hn : ∀ (l : List α), (l.rotate' n).Perm l
        ⊢ (List.nil.rotate' (HAdd.hAdd n 1)).Perm List.nil
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case succ.cons
        α : Type u
        n : Nat
        hn : ∀ (l : List α), (l.rotate' n).Perm l
        hd : α
        tl : List α
        ⊢ ((List.cons hd tl).rotate' (HAdd.hAdd n 1)).Perm (List.cons hd tl)
      -/
    · rw [rotate'_cons_succ]
      /-
        case succ.cons
        α : Type u
        n : Nat
        hn : ∀ (l : List α), (l.rotate' n).Perm l
        hd : α
        tl : List α
        ⊢ ((HAppend.hAppend tl (List.cons hd List.nil)).rotate' n).Perm (List.cons hd  …
      -/
      exact (hn _).trans (perm_append_singleton _ _)
      /-
        🎉 no goals
      -/


@[simp]
theorem nodup_rotate {l : List α} {n : ℕ} : Nodup (l.rotate n) ↔ Nodup l :=
  (rotate_perm l n).nodup_iff


@[simp]
theorem rotate_eq_nil_iff {l : List α} {n : ℕ} : l.rotate n = [] ↔ l = [] := by
  /-
    α : Type u
    l : List α
    n : Nat
    ⊢ Iff (Eq (l.rotate n) List.nil) (Eq l List.nil)
  -/
  induction' n with n hn generalizing l
    /-
      case zero
      α : Type u
      l : List α
      ⊢ Iff (Eq (l.rotate 0) List.nil) (Eq l List.nil)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u
      n : Nat
      hn : ∀ {l : List α}, Iff (Eq (l.rotate n) List.nil) (Eq l List.nil)
      l : List α
      ⊢ Iff (Eq (l.rotate (HAdd.hAdd n 1)) List.nil) (Eq l List.nil)
    -/
  · cases' l with hd tl
      /-
        case succ.nil
        α : Type u
        n : Nat
        hn : ∀ {l : List α}, Iff (Eq (l.rotate n) List.nil) (Eq l List.nil)
        ⊢ Iff (Eq (List.nil.rotate (HAdd.hAdd n 1)) List.nil) (Eq List.nil List.nil)
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case succ.cons
        α : Type u
        n : Nat
        hn : ∀ {l : List α}, Iff (Eq (l.rotate n) List.nil) (Eq l List.nil)
        hd : α
        tl : List α
        ⊢ Iff (Eq ((List.cons hd tl).rotate (HAdd.hAdd n 1)) List.nil) (Eq (List.cons  …
      -/
    · simp [rotate_cons_succ, hn]
      /-
        🎉 no goals
      -/


theorem nil_eq_rotate_iff {l : List α} {n : ℕ} : [] = l.rotate n ↔ [] = l := by
  /-
    α : Type u
    l : List α
    n : Nat
    ⊢ Iff (Eq List.nil (l.rotate n)) (Eq List.nil l)
  -/
  rw [eq_comm, rotate_eq_nil_iff, eq_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem rotate_singleton (x : α) (n : ℕ) : [x].rotate n = [x] :=
  rotate_replicate x 1 n


theorem zipWith_rotate_distrib {β γ : Type*} (f : α → β → γ) (l : List α) (l' : List β) (n : ℕ)
    (h : l.length = l'.length) :
    (zipWith f l l').rotate n = zipWith f (l.rotate n) (l'.rotate n) := by
  rw [rotate_eq_drop_append_take_mod, rotate_eq_drop_append_take_mod,
    rotate_eq_drop_append_take_mod, h, zipWith_append, ← drop_zipWith, ←
    take_zipWith, List.length_zipWith, h, min_self]
  /-
    case h
    α : Type u
    β : Type u_1
    γ : Type u_2
    f : α → β → γ
    l : List α
    l' : List β
    n : Nat
    h : Eq l.length l'.length
    ⊢ Eq (List.drop (HMod.hMod n l'.length) l).length (List.drop (HMod.hMod n l'.l …
  -/
  rw [length_drop, length_drop, h]
  /-
    🎉 no goals
  -/


theorem zipWith_rotate_one {β : Type*} (f : α → α → β) (x y : α) (l : List α) :
    zipWith f (x :: y :: l) ((x :: y :: l).rotate 1) = f x y :: zipWith f (y :: l) (l ++ [x]) := by
  /-
    α : Type u
    β : Type u_1
    f : α → α → β
    x y : α
    l : List α
    ⊢ Eq (List.zipWith f (List.cons x (List.cons y l)) ((List.cons x (List.cons y  …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem getElem?_rotate {l : List α} {n m : ℕ} (hml : m < l.length) :
    (l.rotate n)[m]? = l[(m + n) % l.length]? := by
  /-
    α : Type u
    l : List α
    n m : Nat
    hml : LT.lt m l.length
    ⊢ Eq (GetElem?.getElem? (l.rotate n) m) (GetElem?.getElem? l (HMod.hMod (HAdd. …
  -/
  rw [rotate_eq_drop_append_take_mod]
  /-
    α : Type u
    l : List α
    n m : Nat
    hml : LT.lt m l.length
    ⊢ Eq (GetElem?.getElem? (HAppend.hAppend (List.drop (HMod.hMod n l.length) l)  …
  -/
  rcases lt_or_le m (l.drop (n % l.length)).length with hm | hm
    /-
      case inl
      α : Type u
      l : List α
      n m : Nat
      hml : LT.lt m l.length
      hm : LT.lt m (List.drop (HMod.hMod n l.length) l).length
      ⊢ Eq (GetElem?.getElem? (HAppend.hAppend (List.drop (HMod.hMod n l.length) l)  …
    -/
  · rw [getElem?_append_left hm, getElem?_drop, ← add_mod_mod]
    /-
      case inl
      α : Type u
      l : List α
      n m : Nat
      hml : LT.lt m l.length
      hm : LT.lt m (List.drop (HMod.hMod n l.length) l).length
      ⊢ Eq (GetElem?.getElem? l (HAdd.hAdd (HMod.hMod n l.length) m)) (GetElem?.getE …
    -/
    rw [length_drop, Nat.lt_sub_iff_add_lt] at hm
    /-
      case inl
      α : Type u
      l : List α
      n m : Nat
      hml : LT.lt m l.length
      hm : LT.lt (HAdd.hAdd m (HMod.hMod n l.length)) l.length
      ⊢ Eq (GetElem?.getElem? l (HAdd.hAdd (HMod.hMod n l.length) m)) (GetElem?.getE …
    -/
    rw [mod_eq_of_lt hm, Nat.add_comm]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u
      l : List α
      n m : Nat
      hml : LT.lt m l.length
      hm : LE.le (List.drop (HMod.hMod n l.length) l).length m
      ⊢ Eq (GetElem?.getElem? (HAppend.hAppend (List.drop (HMod.hMod n l.length) l)  …
    -/
  · have hlt : n % length l < length l := mod_lt _ (m.zero_le.trans_lt hml)
    /-
      case inr
      α : Type u
      l : List α
      n m : Nat
      hml : LT.lt m l.length
      hm : LE.le (List.drop (HMod.hMod n l.length) l).length m
      hlt : LT.lt (HMod.hMod n l.length) l.length
      ⊢ Eq (GetElem?.getElem? (HAppend.hAppend (List.drop (HMod.hMod n l.length) l)  …
    -/
    rw [getElem?_append_right hm, getElem?_take_of_lt, length_drop]
      /-
        case inr
        α : Type u
        l : List α
        n m : Nat
        hml : LT.lt m l.length
        hm : LE.le (List.drop (HMod.hMod n l.length) l).length m
        hlt : LT.lt (HMod.hMod n l.length) l.length
        ⊢ Eq (GetElem?.getElem? l (HSub.hSub m (HSub.hSub l.length (HMod.hMod n l.leng …
      -/
    · congr 1
      /-
        case inr.e_a
        α : Type u
        l : List α
        n m : Nat
        hml : LT.lt m l.length
        hm : LE.le (List.drop (HMod.hMod n l.length) l).length m
        hlt : LT.lt (HMod.hMod n l.length) l.length
        ⊢ Eq (HSub.hSub m (HSub.hSub l.length (HMod.hMod n l.length))) (HMod.hMod (HAd …
      -/
      rw [length_drop] at hm
      /-
        case inr.e_a
        α : Type u
        l : List α
        n m : Nat
        hml : LT.lt m l.length
        hm : LE.le (HSub.hSub l.length (HMod.hMod n l.length)) m
        hlt : LT.lt (HMod.hMod n l.length) l.length
        ⊢ Eq (HSub.hSub m (HSub.hSub l.length (HMod.hMod n l.length))) (HMod.hMod (HAd …
      -/
      have hm' := Nat.sub_le_iff_le_add'.1 hm
      have : n % length l + m - length l < length l := by
        rw [Nat.sub_lt_iff_lt_add' hm']
        exact Nat.add_lt_add hlt hml
      /-
        case inr.e_a
        α : Type u
        l : List α
        n m : Nat
        hml : LT.lt m l.length
        hm : LE.le (HSub.hSub l.length (HMod.hMod n l.length)) m
        hlt : LT.lt (HMod.hMod n l.length) l.length
        hm' : LE.le l.length (HAdd.hAdd (HMod.hMod n l.length) m)
        this : LT.lt (HSub.hSub (HAdd.hAdd (HMod.hMod n l.length) m) l.length) l.length
        ⊢ Eq (HSub.hSub m (HSub.hSub l.length (HMod.hMod n l.length))) (HMod.hMod (HAd …
      -/
      conv_rhs => rw [Nat.add_comm m, ← mod_add_mod, mod_eq_sub_mod hm', mod_eq_of_lt this]
      /-
        case inr.e_a
        α : Type u
        l : List α
        n m : Nat
        hml : LT.lt m l.length
        hm : LE.le (HSub.hSub l.length (HMod.hMod n l.length)) m
        hlt : LT.lt (HMod.hMod n l.length) l.length
        hm' : LE.le l.length (HAdd.hAdd (HMod.hMod n l.length) m)
        this : LT.lt (HSub.hSub (HAdd.hAdd (HMod.hMod n l.length) m) l.length) l.length
        ⊢ Eq (HSub.hSub m (HSub.hSub l.length (HMod.hMod n l.length))) (HSub.hSub (HAd …
      -/
      omega
      /-
        🎉 no goals
      -/
      /-
        case inr
        α : Type u
        l : List α
        n m : Nat
        hml : LT.lt m l.length
        hm : LE.le (List.drop (HMod.hMod n l.length) l).length m
        hlt : LT.lt (HMod.hMod n l.length) l.length
        ⊢ LT.lt (HSub.hSub m (List.drop (HMod.hMod n l.length) l).length) (HMod.hMod n …
      -/
    · rwa [Nat.sub_lt_iff_lt_add hm, length_drop, Nat.sub_add_cancel hlt.le]
      /-
        🎉 no goals
      -/


theorem getElem_rotate (l : List α) (n : ℕ) (k : Nat) (h : k < (l.rotate n).length) :
    (l.rotate n)[k] =
      l[(k + n) % l.length]'(mod_lt _ (length_rotate l n ▸ k.zero_le.trans_lt h)) := by
  /-
    α : Type u
    l : List α
    n k : Nat
    h : LT.lt k (l.rotate n).length
    ⊢ Eq (GetElem.getElem (l.rotate n) k h) (GetElem.getElem l (HMod.hMod (HAdd.hA …
  -/
  rw [← Option.some_inj, ← getElem?_eq_getElem, ← getElem?_eq_getElem, getElem?_rotate]
  /-
    α : Type u
    l : List α
    n k : Nat
    h : LT.lt k (l.rotate n).length
    ⊢ LT.lt k l.length
  -/
  exact h.trans_eq (length_rotate _ _)
  /-
    🎉 no goals
  -/


theorem get?_rotate {l : List α} {n m : ℕ} (hml : m < l.length) :
    (l.rotate n).get? m = l.get? ((m + n) % l.length) := by
  /-
    α : Type u
    l : List α
    n m : Nat
    hml : LT.lt m l.length
    ⊢ Eq ((l.rotate n).get? m) (l.get? (HMod.hMod (HAdd.hAdd m n) l.length))
  -/
  simp only [get?_eq_getElem?, length_rotate, hml, getElem?_eq_getElem, getElem_rotate]
  /-
    α : Type u
    l : List α
    n m : Nat
    hml : LT.lt m l.length
    ⊢ Eq (Option.some (GetElem.getElem l (HMod.hMod (HAdd.hAdd m n) l.length) ⋯))  …
  -/
  rw [← getElem?_eq_getElem]
  /-
    🎉 no goals
  -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10756): new lemma

theorem get_rotate (l : List α) (n : ℕ) (k : Fin (l.rotate n).length) :
    (l.rotate n).get k = l.get ⟨(k + n) % l.length, mod_lt _ (length_rotate l n ▸ k.pos)⟩ := by
  /-
    α : Type u
    l : List α
    n : Nat
    k : Fin (l.rotate n).length
    ⊢ Eq ((l.rotate n).get k) (l.get ⟨HMod.hMod (HAdd.hAdd (↑k) n) l.length, ⋯⟩)
  -/
  simp [getElem_rotate]
  /-
    🎉 no goals
  -/


theorem head?_rotate {l : List α} {n : ℕ} (h : n < l.length) : head? (l.rotate n) = l[n]? := by
  rw [← get?_zero, get?_rotate (n.zero_le.trans_lt h), Nat.zero_add, Nat.mod_eq_of_lt h,
    get?_eq_getElem?]


theorem get_rotate_one (l : List α) (k : Fin (l.rotate 1).length) :
    (l.rotate 1).get k = l.get ⟨(k + 1) % l.length, mod_lt _ (length_rotate l 1 ▸ k.pos)⟩ :=
  get_rotate l 1 k


@[deprecated (since := "2024-08-19")] alias nthLe_rotate_one := get_rotate_one


/-- A version of `List.get_rotate` that represents `List.get l` in terms of
`List.get (List.rotate l n)`, not vice versa. Can be used instead of rewriting `List.get_rotate`
from right to left. -/
theorem get_eq_get_rotate (l : List α) (n : ℕ) (k : Fin l.length) :
    l.get k = (l.rotate n).get ⟨(l.length - n % l.length + k) % l.length,
      (Nat.mod_lt _ (k.1.zero_le.trans_lt k.2)).trans_eq (length_rotate _ _).symm⟩ := by
  /-
    α : Type u
    l : List α
    n : Nat
    k : Fin l.length
    ⊢ Eq (l.get k) ((l.rotate n).get ⟨HMod.hMod (HAdd.hAdd (HSub.hSub l.length (HM …
  -/
  rw [get_rotate]
  /-
    α : Type u
    l : List α
    n : Nat
    k : Fin l.length
    ⊢ Eq (l.get k) (l.get ⟨HMod.hMod (HAdd.hAdd (↑⟨HMod.hMod (HAdd.hAdd (HSub.hSub …
  -/
  refine congr_arg l.get (Fin.eq_of_val_eq ?_)
  /-
    α : Type u
    l : List α
    n : Nat
    k : Fin l.length
    ⊢ Eq ↑k ↑⟨HMod.hMod (HAdd.hAdd (↑⟨HMod.hMod (HAdd.hAdd (HSub.hSub l.length (HM …
  -/
  simp only [mod_add_mod]
  /-
    α : Type u
    l : List α
    n : Nat
    k : Fin l.length
    ⊢ Eq (↑k) (HMod.hMod (HAdd.hAdd (HAdd.hAdd (HSub.hSub l.length (HMod.hMod n l. …
  -/
  rw [← add_mod_mod, Nat.add_right_comm, Nat.sub_add_cancel, add_mod_left, mod_eq_of_lt]
  /-
    α : Type u
    l : List α
    n : Nat
    k : Fin l.length
    ⊢ LT.lt (↑k) l.length
  -/
  exacts [k.2, (mod_lt _ (k.1.zero_le.trans_lt k.2)).le]
  /-
    🎉 no goals
  -/


theorem rotate_eq_self_iff_eq_replicate [hα : Nonempty α] :
    ∀ {l : List α}, (∀ n, l.rotate n = l) ↔ ∃ a, l = replicate l.length a
             /-
               α : Type u
               hα : Nonempty α
               ⊢ Iff (∀ (n : Nat), Eq (List.nil.rotate n) List.nil) (Exists fun a => Eq List. …
             -/
  | [] => by simp
             /-
               🎉 no goals
             -/
  | a :: l => ⟨fun h => ⟨a, ext_getElem (length_replicate _ _).symm fun n h₁ h₂ => by
      rw [getElem_replicate, ← Option.some_inj, ← getElem?_eq_getElem, ← head?_rotate h₁, h,
        head?_cons]⟩,
                        /-
                          α : Type u
                          hα : Nonempty α
                          a : α
                          l : List α
                          x✝ : Exists fun a_1 => Eq (List.cons a l) (List.replicate (List.cons a l).leng …
                          n : Nat
                          b : α
                          hb : Eq (List.cons a l) (List.replicate (List.cons a l).length b)
                          ⊢ Eq ((List.cons a l).rotate n) (List.cons a l)
                        -/
    fun ⟨b, hb⟩ n => by rw [hb, rotate_replicate]⟩
                        /-
                          🎉 no goals
                        -/


theorem rotate_one_eq_self_iff_eq_replicate [Nonempty α] {l : List α} :
    l.rotate 1 = l ↔ ∃ a : α, l = List.replicate l.length a :=
  ⟨fun h =>
    rotate_eq_self_iff_eq_replicate.mp fun n =>
                                            /-
                                              α : Type u
                                              inst✝ : Nonempty α
                                              l : List α
                                              h : Eq (l.rotate 1) l
                                              n✝ n : Nat
                                              hn : Eq (l.rotate n) l
                                              ⊢ Eq (l.rotate n.succ) l
                                            -/
      Nat.rec l.rotate_zero (fun n hn => by rwa [Nat.succ_eq_add_one, ← l.rotate_rotate, hn]) n,
                                            /-
                                              🎉 no goals
                                            -/
    fun h => rotate_eq_self_iff_eq_replicate.mpr h 1⟩


theorem rotate_injective (n : ℕ) : Function.Injective fun l : List α => l.rotate n := by
  /-
    α : Type u
    n : Nat
    ⊢ Function.Injective fun l => l.rotate n
  -/
  rintro l l' (h : l.rotate n = l'.rotate n)
  /-
    α : Type u
    n : Nat
    l l' : List α
    h : Eq (l.rotate n) (l'.rotate n)
    ⊢ Eq l l'
  -/
  have hle : l.length = l'.length := (l.length_rotate n).symm.trans (h.symm ▸ l'.length_rotate n)
  /-
    α : Type u
    n : Nat
    l l' : List α
    h : Eq (l.rotate n) (l'.rotate n)
    hle : Eq l.length l'.length
    ⊢ Eq l l'
  -/
  rw [rotate_eq_drop_append_take_mod, rotate_eq_drop_append_take_mod] at h
  /-
    α : Type u
    n : Nat
    l l' : List α
    h : Eq (HAppend.hAppend (List.drop (HMod.hMod n l.length) l) (List.take (HMod. …
    hle : Eq l.length l'.length
    ⊢ Eq l l'
  -/
  obtain ⟨hd, ht⟩ := append_inj h (by simp_all)
  /-
    case intro
    α : Type u
    n : Nat
    l l' : List α
    h : Eq (HAppend.hAppend (List.drop (HMod.hMod n l.length) l) (List.take (HMod. …
    hle : Eq l.length l'.length
    hd : Eq (List.drop (HMod.hMod n l.length) l) (List.drop (HMod.hMod n l'.length …
    ht : Eq (List.take (HMod.hMod n l.length) l) (List.take (HMod.hMod n l'.length …
    ⊢ Eq l l'
  -/
  rw [← take_append_drop _ l, ht, hd, take_append_drop]
  /-
    🎉 no goals
  -/


@[simp]
theorem rotate_eq_rotate {l l' : List α} {n : ℕ} : l.rotate n = l'.rotate n ↔ l = l' :=
  (rotate_injective n).eq_iff


theorem rotate_eq_iff {l l' : List α} {n : ℕ} :
    l.rotate n = l' ↔ l = l'.rotate (l'.length - n % l'.length) := by
  /-
    α : Type u
    l l' : List α
    n : Nat
    ⊢ Iff (Eq (l.rotate n) l') (Eq l (l'.rotate (HSub.hSub l'.length (HMod.hMod n  …
  -/
  rw [← @rotate_eq_rotate _ l _ n, rotate_rotate, ← rotate_mod l', add_mod]
  /-
    α : Type u
    l l' : List α
    n : Nat
    ⊢ Iff (Eq (l.rotate n) l') (Eq (l.rotate n) (l'.rotate (HMod.hMod (HAdd.hAdd ( …
  -/
  rcases l'.length.zero_le.eq_or_lt with hl | hl
    /-
      case inl
      α : Type u
      l l' : List α
      n : Nat
      hl : Eq 0 l'.length
      ⊢ Iff (Eq (l.rotate n) l') (Eq (l.rotate n) (l'.rotate (HMod.hMod (HAdd.hAdd ( …
    -/
  · rw [eq_nil_of_length_eq_zero hl.symm, rotate_nil]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u
      l l' : List α
      n : Nat
      hl : LT.lt 0 l'.length
      ⊢ Iff (Eq (l.rotate n) l') (Eq (l.rotate n) (l'.rotate (HMod.hMod (HAdd.hAdd ( …
    -/
  · rcases (Nat.zero_le (n % l'.length)).eq_or_lt with hn | hn
      /-
        case inr.inl
        α : Type u
        l l' : List α
        n : Nat
        hl : LT.lt 0 l'.length
        hn : Eq 0 (HMod.hMod n l'.length)
        ⊢ Iff (Eq (l.rotate n) l') (Eq (l.rotate n) (l'.rotate (HMod.hMod (HAdd.hAdd ( …
      -/
    · simp [← hn]
      /-
        🎉 no goals
      -/
      /-
        case inr.inr
        α : Type u
        l l' : List α
        n : Nat
        hl : LT.lt 0 l'.length
        hn : LT.lt 0 (HMod.hMod n l'.length)
        ⊢ Iff (Eq (l.rotate n) l') (Eq (l.rotate n) (l'.rotate (HMod.hMod (HAdd.hAdd ( …
      -/
    · rw [mod_eq_of_lt (Nat.sub_lt hl hn), Nat.sub_add_cancel, mod_self, rotate_zero]
      /-
        case inr.inr
        α : Type u
        l l' : List α
        n : Nat
        hl : LT.lt 0 l'.length
        hn : LT.lt 0 (HMod.hMod n l'.length)
        ⊢ LE.le (HMod.hMod n l'.length) l'.length
      -/
      exact (Nat.mod_lt _ hl).le
      /-
        🎉 no goals
      -/


@[simp]
theorem rotate_eq_singleton_iff {l : List α} {n : ℕ} {x : α} : l.rotate n = [x] ↔ l = [x] := by
  /-
    α : Type u
    l : List α
    n : Nat
    x : α
    ⊢ Iff (Eq (l.rotate n) (List.cons x List.nil)) (Eq l (List.cons x List.nil))
  -/
  rw [rotate_eq_iff, rotate_singleton]
  /-
    🎉 no goals
  -/


@[simp]
theorem singleton_eq_rotate_iff {l : List α} {n : ℕ} {x : α} : [x] = l.rotate n ↔ [x] = l := by
  /-
    α : Type u
    l : List α
    n : Nat
    x : α
    ⊢ Iff (Eq (List.cons x List.nil) (l.rotate n)) (Eq (List.cons x List.nil) l)
  -/
  rw [eq_comm, rotate_eq_singleton_iff, eq_comm]
  /-
    🎉 no goals
  -/


theorem reverse_rotate (l : List α) (n : ℕ) :
    (l.rotate n).reverse = l.reverse.rotate (l.length - n % l.length) := by
  /-
    α : Type u
    l : List α
    n : Nat
    ⊢ Eq (l.rotate n).reverse (l.reverse.rotate (HSub.hSub l.length (HMod.hMod n l …
  -/
  rw [← length_reverse l, ← rotate_eq_iff]
  /-
    α : Type u
    l : List α
    n : Nat
    ⊢ Eq ((l.rotate n).reverse.rotate n) l.reverse
  -/
  induction' n with n hn generalizing l
    /-
      case zero
      α : Type u
      l : List α
      ⊢ Eq ((l.rotate 0).reverse.rotate 0) l.reverse
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u
      n : Nat
      hn : ∀ (l : List α), Eq ((l.rotate n).reverse.rotate n) l.reverse
      l : List α
      ⊢ Eq ((l.rotate (HAdd.hAdd n 1)).reverse.rotate (HAdd.hAdd n 1)) l.reverse
    -/
  · cases' l with hd tl
      /-
        case succ.nil
        α : Type u
        n : Nat
        hn : ∀ (l : List α), Eq ((l.rotate n).reverse.rotate n) l.reverse
        ⊢ Eq ((List.nil.rotate (HAdd.hAdd n 1)).reverse.rotate (HAdd.hAdd n 1)) List.n …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case succ.cons
        α : Type u
        n : Nat
        hn : ∀ (l : List α), Eq ((l.rotate n).reverse.rotate n) l.reverse
        hd : α
        tl : List α
        ⊢ Eq (((List.cons hd tl).rotate (HAdd.hAdd n 1)).reverse.rotate (HAdd.hAdd n 1 …
      -/
    · rw [rotate_cons_succ, ← rotate_rotate, hn]
      /-
        case succ.cons
        α : Type u
        n : Nat
        hn : ∀ (l : List α), Eq ((l.rotate n).reverse.rotate n) l.reverse
        hd : α
        tl : List α
        ⊢ Eq ((HAppend.hAppend tl (List.cons hd List.nil)).reverse.rotate 1) (List.con …
      -/
      simp
      /-
        🎉 no goals
      -/


theorem rotate_reverse (l : List α) (n : ℕ) :
    l.reverse.rotate n = (l.rotate (l.length - n % l.length)).reverse := by
  /-
    α : Type u
    l : List α
    n : Nat
    ⊢ Eq (l.reverse.rotate n) (l.rotate (HSub.hSub l.length (HMod.hMod n l.length) …
  -/
  rw [← reverse_reverse l]
  simp_rw [reverse_rotate, reverse_reverse, rotate_eq_iff, rotate_rotate, length_rotate,
    length_reverse]
  /-
    α : Type u
    l : List α
    n : Nat
    ⊢ Eq l.reverse (l.reverse.rotate (HAdd.hAdd (HSub.hSub l.length (HMod.hMod (HS …
  -/
  rw [← length_reverse l]
  /-
    α : Type u
    l : List α
    n : Nat
    ⊢ Eq l.reverse (l.reverse.rotate (HAdd.hAdd (HSub.hSub l.reverse.length (HMod. …
  -/
  let k := n % l.reverse.length
  /-
    α : Type u
    l : List α
    n : Nat
    k : Nat := HMod.hMod n l.reverse.length
    ⊢ Eq l.reverse (l.reverse.rotate (HAdd.hAdd (HSub.hSub l.reverse.length (HMod. …
  -/
  cases' hk' : k with k'
    /-
      case zero
      α : Type u
      l : List α
      n : Nat
      k : Nat := HMod.hMod n l.reverse.length
      hk' : Eq k 0
      ⊢ Eq l.reverse (l.reverse.rotate (HAdd.hAdd (HSub.hSub l.reverse.length (HMod. …
    -/
  · simp_all! [k, length_reverse, ← rotate_rotate]
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u
      l : List α
      n : Nat
      k : Nat := HMod.hMod n l.reverse.length
      k' : Nat
      hk' : Eq k (HAdd.hAdd k' 1)
      ⊢ Eq l.reverse (l.reverse.rotate (HAdd.hAdd (HSub.hSub l.reverse.length (HMod. …
    -/
  · cases' l with x l
      /-
        case succ.nil
        α : Type u
        n k' : Nat
        k : Nat := HMod.hMod n List.nil.reverse.length
        hk' : Eq k (HAdd.hAdd k' 1)
        ⊢ Eq List.nil.reverse (List.nil.reverse.rotate (HAdd.hAdd (HSub.hSub List.nil. …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case succ.cons
        α : Type u
        n k' : Nat
        x : α
        l : List α
        k : Nat := HMod.hMod n (List.cons x l).reverse.length
        hk' : Eq k (HAdd.hAdd k' 1)
        ⊢ Eq (List.cons x l).reverse ((List.cons x l).reverse.rotate (HAdd.hAdd (HSub. …
      -/
    · rw [Nat.mod_eq_of_lt, Nat.sub_add_cancel, rotate_length]
        /-
          case succ.cons
          α : Type u
          n k' : Nat
          x : α
          l : List α
          k : Nat := HMod.hMod n (List.cons x l).reverse.length
          hk' : Eq k (HAdd.hAdd k' 1)
          ⊢ LE.le (HSub.hSub (List.cons x l).reverse.length (HMod.hMod n (List.cons x l) …
        -/
      · exact Nat.sub_le _ _
        /-
          🎉 no goals
        -/
        /-
          case succ.cons
          α : Type u
          n k' : Nat
          x : α
          l : List α
          k : Nat := HMod.hMod n (List.cons x l).reverse.length
          hk' : Eq k (HAdd.hAdd k' 1)
          ⊢ LT.lt (HSub.hSub (List.cons x l).reverse.length (HMod.hMod n (List.cons x l) …
        -/
      · exact Nat.sub_lt (by simp) (by simp_all! [k])
        /-
          🎉 no goals
        -/


theorem map_rotate {β : Type*} (f : α → β) (l : List α) (n : ℕ) :
    map f (l.rotate n) = (map f l).rotate n := by
  /-
    α : Type u
    β : Type u_1
    f : α → β
    l : List α
    n : Nat
    ⊢ Eq (List.map f (l.rotate n)) ((List.map f l).rotate n)
  -/
  induction' n with n hn IH generalizing l
    /-
      case zero
      α : Type u
      β : Type u_1
      f : α → β
      l : List α
      ⊢ Eq (List.map f (l.rotate 0)) ((List.map f l).rotate 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u
      β : Type u_1
      f : α → β
      n : Nat
      hn : ∀ (l : List α), Eq (List.map f (l.rotate n)) ((List.map f l).rotate n)
      l : List α
      ⊢ Eq (List.map f (l.rotate (HAdd.hAdd n 1))) ((List.map f l).rotate (HAdd.hAdd …
    -/
  · cases' l with hd tl
      /-
        case succ.nil
        α : Type u
        β : Type u_1
        f : α → β
        n : Nat
        hn : ∀ (l : List α), Eq (List.map f (l.rotate n)) ((List.map f l).rotate n)
        ⊢ Eq (List.map f (List.nil.rotate (HAdd.hAdd n 1))) ((List.map f List.nil).rot …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case succ.cons
        α : Type u
        β : Type u_1
        f : α → β
        n : Nat
        hn : ∀ (l : List α), Eq (List.map f (l.rotate n)) ((List.map f l).rotate n)
        hd : α
        tl : List α
        ⊢ Eq (List.map f ((List.cons hd tl).rotate (HAdd.hAdd n 1))) ((List.map f (Lis …
      -/
    · simp [hn]
      /-
        🎉 no goals
      -/


theorem Nodup.rotate_congr {l : List α} (hl : l.Nodup) (hn : l ≠ []) (i j : ℕ)
    (h : l.rotate i = l.rotate j) : i % l.length = j % l.length := by
  /-
    α : Type u
    l : List α
    hl : l.Nodup
    hn : Ne l List.nil
    i j : Nat
    h : Eq (l.rotate i) (l.rotate j)
    ⊢ Eq (HMod.hMod i l.length) (HMod.hMod j l.length)
  -/
  rw [← rotate_mod l i, ← rotate_mod l j] at h
  simpa only [head?_rotate, mod_lt, length_pos_of_ne_nil hn, getElem?_eq_getElem, Option.some_inj,
    hl.getElem_inj_iff, Fin.ext_iff] using congr_arg head? h


theorem Nodup.rotate_congr_iff {l : List α} (hl : l.Nodup) {i j : ℕ} :
    l.rotate i = l.rotate j ↔ i % l.length = j % l.length ∨ l = [] := by
  /-
    α : Type u
    l : List α
    hl : l.Nodup
    i j : Nat
    ⊢ Iff (Eq (l.rotate i) (l.rotate j)) (Or (Eq (HMod.hMod i l.length) (HMod.hMod …
  -/
  rcases eq_or_ne l [] with rfl | hn
    /-
      case inl
      α : Type u
      i j : Nat
      hl : List.nil.Nodup
      ⊢ Iff (Eq (List.nil.rotate i) (List.nil.rotate j)) (Or (Eq (HMod.hMod i List.n …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u
      l : List α
      hl : l.Nodup
      i j : Nat
      hn : Ne l List.nil
      ⊢ Iff (Eq (l.rotate i) (l.rotate j)) (Or (Eq (HMod.hMod i l.length) (HMod.hMod …
    -/
  · simp only [hn, or_false]
    /-
      case inr
      α : Type u
      l : List α
      hl : l.Nodup
      i j : Nat
      hn : Ne l List.nil
      ⊢ Iff (Eq (l.rotate i) (l.rotate j)) (Eq (HMod.hMod i l.length) (HMod.hMod j l …
    -/
    refine ⟨hl.rotate_congr hn _ _, fun h ↦ ?_⟩
    /-
      case inr
      α : Type u
      l : List α
      hl : l.Nodup
      i j : Nat
      hn : Ne l List.nil
      h : Eq (HMod.hMod i l.length) (HMod.hMod j l.length)
      ⊢ Eq (l.rotate i) (l.rotate j)
    -/
    rw [← rotate_mod, h, rotate_mod]
    /-
      🎉 no goals
    -/


theorem Nodup.rotate_eq_self_iff {l : List α} (hl : l.Nodup) {n : ℕ} :
    l.rotate n = l ↔ n % l.length = 0 ∨ l = [] := by
  /-
    α : Type u
    l : List α
    hl : l.Nodup
    n : Nat
    ⊢ Iff (Eq (l.rotate n) l) (Or (Eq (HMod.hMod n l.length) 0) (Eq l List.nil))
  -/
  rw [← zero_mod, ← hl.rotate_congr_iff, rotate_zero]
  /-
    🎉 no goals
  -/


/-- `IsRotated l₁ l₂` or `l₁ ~r l₂` asserts that `l₁` and `l₂` are cyclic permutations
  of each other. This is defined by claiming that `∃ n, l.rotate n = l'`. -/
def IsRotated : Prop :=
  ∃ n, l.rotate n = l'


@[inherit_doc List.IsRotated]
-- This matches the precedence of the infix `~` for `List.Perm`, and of other relation infixes
infixr:50 " ~r " => IsRotated


@[refl]
theorem IsRotated.refl (l : List α) : l ~r l :=
         /-
           α : Type u
           l : List α
           ⊢ Eq (l.rotate 0) l
         -/
  ⟨0, by simp⟩
         /-
           🎉 no goals
         -/


@[symm]
theorem IsRotated.symm (h : l ~r l') : l' ~r l := by
  /-
    α : Type u
    l l' : List α
    h : l.IsRotated l'
    ⊢ l'.IsRotated l
  -/
  obtain ⟨n, rfl⟩ := h
  /-
    case intro
    α : Type u
    l : List α
    n : Nat
    ⊢ (l.rotate n).IsRotated l
  -/
  cases' l with hd tl
    /-
      case intro.nil
      α : Type u
      n : Nat
      ⊢ (List.nil.rotate n).IsRotated List.nil
    -/
  · exists 0
    /-
      🎉 no goals
    -/
    /-
      case intro.cons
      α : Type u
      n : Nat
      hd : α
      tl : List α
      ⊢ ((List.cons hd tl).rotate n).IsRotated (List.cons hd tl)
    -/
  · use (hd :: tl).length * n - n
    /-
      case h
      α : Type u
      n : Nat
      hd : α
      tl : List α
      ⊢ Eq (((List.cons hd tl).rotate n).rotate (HSub.hSub (HMul.hMul (List.cons hd  …
    -/
    rw [rotate_rotate, Nat.add_sub_cancel', rotate_length_mul]
    /-
      case h
      α : Type u
      n : Nat
      hd : α
      tl : List α
      ⊢ LE.le n (HMul.hMul (List.cons hd tl).length n)
    -/
    exact Nat.le_mul_of_pos_left _ (by simp)
    /-
      🎉 no goals
    -/


theorem isRotated_comm : l ~r l' ↔ l' ~r l :=
  ⟨IsRotated.symm, IsRotated.symm⟩


@[simp]
protected theorem IsRotated.forall (l : List α) (n : ℕ) : l.rotate n ~r l :=
  IsRotated.symm ⟨n, rfl⟩


@[trans]
theorem IsRotated.trans : ∀ {l l' l'' : List α}, l ~r l' → l' ~r l'' → l ~r l''
                                              /-
                                                α : Type u
                                                l✝ : List α
                                                n m : Nat
                                                ⊢ Eq (l✝.rotate (HAdd.hAdd n m)) ((l✝.rotate n).rotate m)
                                              -/
  | _, _, _, ⟨n, rfl⟩, ⟨m, rfl⟩ => ⟨n + m, by rw [rotate_rotate]⟩
                                              /-
                                                🎉 no goals
                                              -/


theorem IsRotated.eqv : Equivalence (@IsRotated α) :=
  Equivalence.mk IsRotated.refl IsRotated.symm IsRotated.trans


/-- The relation `List.IsRotated l l'` forms a `Setoid` of cycles. -/
def IsRotated.setoid (α : Type*) : Setoid (List α) where
  r := IsRotated
  iseqv := IsRotated.eqv


theorem IsRotated.perm (h : l ~r l') : l ~ l' :=
  Exists.elim h fun _ hl => hl ▸ (rotate_perm _ _).symm


theorem IsRotated.nodup_iff (h : l ~r l') : Nodup l ↔ Nodup l' :=
  h.perm.nodup_iff


theorem IsRotated.mem_iff (h : l ~r l') {a : α} : a ∈ l ↔ a ∈ l' :=
  h.perm.mem_iff


@[simp]
theorem isRotated_nil_iff : l ~r [] ↔ l = [] :=
                     /-
                       α : Type u
                       l : List α
                       x✝ : l.IsRotated List.nil
                       n : Nat
                       hn : Eq (l.rotate n) List.nil
                       ⊢ Eq l List.nil
                     -/
                     /-
                       🎉 no goals
                     -/
  ⟨fun ⟨n, hn⟩ => by simpa using hn, fun h => h ▸ by rfl⟩
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
theorem isRotated_nil_iff' : [] ~r l ↔ [] = l := by
  /-
    α : Type u
    l : List α
    ⊢ Iff (List.nil.IsRotated l) (Eq List.nil l)
  -/
  rw [isRotated_comm, isRotated_nil_iff, eq_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem isRotated_singleton_iff {x : α} : l ~r [x] ↔ l = [x] :=
                     /-
                       α : Type u
                       l : List α
                       x : α
                       x✝ : l.IsRotated (List.cons x List.nil)
                       n : Nat
                       hn : Eq (l.rotate n) (List.cons x List.nil)
                       ⊢ Eq l (List.cons x List.nil)
                     -/
                     /-
                       🎉 no goals
                     -/
  ⟨fun ⟨n, hn⟩ => by simpa using hn, fun h => h ▸ by rfl⟩
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
theorem isRotated_singleton_iff' {x : α} : [x] ~r l ↔ [x] = l := by
  /-
    α : Type u
    l : List α
    x : α
    ⊢ Iff ((List.cons x List.nil).IsRotated l) (Eq (List.cons x List.nil) l)
  -/
  rw [isRotated_comm, isRotated_singleton_iff, eq_comm]
  /-
    🎉 no goals
  -/


theorem isRotated_concat (hd : α) (tl : List α) : (tl ++ [hd]) ~r (hd :: tl) :=
                        /-
                          α : Type u
                          hd : α
                          tl : List α
                          ⊢ Eq ((List.cons hd tl).rotate 1) (HAppend.hAppend tl (List.cons hd List.nil))
                        -/
  IsRotated.symm ⟨1, by simp⟩
                        /-
                          🎉 no goals
                        -/


theorem isRotated_append : (l ++ l') ~r (l' ++ l) :=
                /-
                  α : Type u
                  l l' : List α
                  ⊢ Eq ((HAppend.hAppend l l').rotate l.length) (HAppend.hAppend l' l)
                -/
  ⟨l.length, by simp⟩
                /-
                  🎉 no goals
                -/


theorem IsRotated.reverse (h : l ~r l') : l.reverse ~r l'.reverse := by
  /-
    α : Type u
    l l' : List α
    h : l.IsRotated l'
    ⊢ l.reverse.IsRotated l'.reverse
  -/
  obtain ⟨n, rfl⟩ := h
  /-
    case intro
    α : Type u
    l : List α
    n : Nat
    ⊢ l.reverse.IsRotated (l.rotate n).reverse
  -/
  exact ⟨_, (reverse_rotate _ _).symm⟩
  /-
    🎉 no goals
  -/


theorem isRotated_reverse_comm_iff : l.reverse ~r l' ↔ l ~r l'.reverse := by
  /-
    α : Type u
    l l' : List α
    ⊢ Iff (l.reverse.IsRotated l') (l.IsRotated l'.reverse)
  -/
  constructor <;>
      /-
        case mp
        α : Type u
        l l' : List α
        ⊢ l.reverse.IsRotated l' → l.IsRotated l'.reverse
      -/
      /-
        case mp
        α : Type u
        l l' : List α
        h : l.reverse.IsRotated l'
        ⊢ l.IsRotated l'.reverse
      -/
      /-
        🎉 no goals
      -/
      /-
        case mpr
        α : Type u
        l l' : List α
        h : l.IsRotated l'.reverse
        ⊢ l.reverse.IsRotated l'
      -/
      simpa using h.reverse
      /-
        🎉 no goals
      -/


@[simp]
theorem isRotated_reverse_iff : l.reverse ~r l'.reverse ↔ l ~r l' := by
  /-
    α : Type u
    l l' : List α
    ⊢ Iff (l.reverse.IsRotated l'.reverse) (l.IsRotated l')
  -/
  simp [isRotated_reverse_comm_iff]
  /-
    🎉 no goals
  -/


theorem isRotated_iff_mod : l ~r l' ↔ ∃ n ≤ l.length, l.rotate n = l' := by
  /-
    α : Type u
    l l' : List α
    ⊢ Iff (l.IsRotated l') (Exists fun n => And (LE.le n l.length) (Eq (l.rotate n …
  -/
  refine ⟨fun h => ?_, fun ⟨n, _, h⟩ => ⟨n, h⟩⟩
  /-
    α : Type u
    l l' : List α
    h : l.IsRotated l'
    ⊢ Exists fun n => And (LE.le n l.length) (Eq (l.rotate n) l')
  -/
  obtain ⟨n, rfl⟩ := h
  /-
    case intro
    α : Type u
    l : List α
    n : Nat
    ⊢ Exists fun n_1 => And (LE.le n_1 l.length) (Eq (l.rotate n_1) (l.rotate n))
  -/
  cases' l with hd tl
    /-
      case intro.nil
      α : Type u
      n : Nat
      ⊢ Exists fun n_1 => And (LE.le n_1 List.nil.length) (Eq (List.nil.rotate n_1)  …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case intro.cons
      α : Type u
      n : Nat
      hd : α
      tl : List α
      ⊢ Exists fun n_1 => And (LE.le n_1 (List.cons hd tl).length) (Eq ((List.cons h …
    -/
  · refine ⟨n % (hd :: tl).length, ?_, rotate_mod _ _⟩
    /-
      case intro.cons
      α : Type u
      n : Nat
      hd : α
      tl : List α
      ⊢ LE.le (HMod.hMod n (List.cons hd tl).length) (List.cons hd tl).length
    -/
    refine (Nat.mod_lt _ ?_).le
    /-
      case intro.cons
      α : Type u
      n : Nat
      hd : α
      tl : List α
      ⊢ GT.gt (List.cons hd tl).length 0
    -/
    simp
    /-
      🎉 no goals
    -/


theorem isRotated_iff_mem_map_range : l ~r l' ↔ l' ∈ (List.range (l.length + 1)).map l.rotate := by
  /-
    α : Type u
    l l' : List α
    ⊢ Iff (l.IsRotated l') (Membership.mem (List.map l.rotate (List.range (HAdd.hA …
  -/
  simp_rw [mem_map, mem_range, isRotated_iff_mod]
  exact
    ⟨fun ⟨n, hn, h⟩ => ⟨n, Nat.lt_succ_of_le hn, h⟩,
      fun ⟨n, hn, h⟩ => ⟨n, Nat.le_of_lt_succ hn, h⟩⟩

-- Porting note: @[congr] only works for equality.
-- @[congr]

theorem IsRotated.map {β : Type*} {l₁ l₂ : List α} (h : l₁ ~r l₂) (f : α → β) :
    map f l₁ ~r map f l₂ := by
  /-
    α : Type u
    β : Type u_1
    l₁ l₂ : List α
    h : l₁.IsRotated l₂
    f : α → β
    ⊢ (List.map f l₁).IsRotated (List.map f l₂)
  -/
  obtain ⟨n, rfl⟩ := h
  /-
    case intro
    α : Type u
    β : Type u_1
    l₁ : List α
    f : α → β
    n : Nat
    ⊢ (List.map f l₁).IsRotated (List.map f (l₁.rotate n))
  -/
  rw [map_rotate]
  /-
    case intro
    α : Type u
    β : Type u_1
    l₁ : List α
    f : α → β
    n : Nat
    ⊢ (List.map f l₁).IsRotated ((List.map f l₁).rotate n)
  -/
  use n
  /-
    🎉 no goals
  -/


theorem IsRotated.cons_getLast_dropLast
    (L : List α) (hL : L ≠ []) : L.getLast hL :: L.dropLast ~r L := by
  induction L using List.reverseRecOn with
  | nil => simp at hL
  | append_singleton a L _ =>
    simp only [getLast_append, dropLast_concat]
    apply IsRotated.symm
    apply isRotated_concat


theorem IsRotated.dropLast_tail {α}
    {L : List α} (hL : L ≠ []) (hL' : L.head hL = L.getLast hL) : L.dropLast ~r L.tail :=
  match L with
             /-
               α : Type u_1
               L : List α
               hL : Ne List.nil List.nil
               hL' : Eq (List.nil.head hL) (List.nil.getLast hL)
               ⊢ List.nil.dropLast.IsRotated List.nil.tail
             -/
  | [] => by simp
             /-
               🎉 no goals
             -/
              /-
                α : Type u_1
                L : List α
                head✝ : α
                hL : Ne (List.cons head✝ List.nil) List.nil
                hL' : Eq ((List.cons head✝ List.nil).head hL) ((List.cons head✝ List.nil).getL …
                ⊢ (List.cons head✝ List.nil).dropLast.IsRotated (List.cons head✝ List.nil).tail
              -/
  | [_] => by simp
              /-
                🎉 no goals
              -/
  | a :: b :: L => by
    /-
      α : Type u_1
      L✝ : List α
      a b : α
      L : List α
      hL : Ne (List.cons a (List.cons b L)) List.nil
      hL' : Eq ((List.cons a (List.cons b L)).head hL) ((List.cons a (List.cons b L) …
      ⊢ (List.cons a (List.cons b L)).dropLast.IsRotated (List.cons a (List.cons b L …
    -/
    simp only [head_cons, ne_eq, reduceCtorEq, not_false_eq_true, getLast_cons] at hL'
    /-
      α : Type u_1
      L✝ : List α
      a b : α
      L : List α
      hL : Ne (List.cons a (List.cons b L)) List.nil
      hL' : Eq a ((List.cons b L).getLast ⋯)
      ⊢ (List.cons a (List.cons b L)).dropLast.IsRotated (List.cons a (List.cons b L …
    -/
    simp [hL', IsRotated.cons_getLast_dropLast]
    /-
      🎉 no goals
    -/


/-- List of all cyclic permutations of `l`.
The `cyclicPermutations` of a nonempty list `l` will always contain `List.length l` elements.
This implies that under certain conditions, there are duplicates in `List.cyclicPermutations l`.
The `n`th entry is equal to `l.rotate n`, proven in `List.get_cyclicPermutations`.
The proof that every cyclic permutant of `l` is in the list is `List.mem_cyclicPermutations_iff`.

     cyclicPermutations [1, 2, 3, 2, 4] =
       [[1, 2, 3, 2, 4], [2, 3, 2, 4, 1], [3, 2, 4, 1, 2],
        [2, 4, 1, 2, 3], [4, 1, 2, 3, 2]] -/
def cyclicPermutations : List α → List (List α)
  | [] => [[]]
  | l@(_ :: _) => dropLast (zipWith (· ++ ·) (tails l) (inits l))


@[simp]
theorem cyclicPermutations_nil : cyclicPermutations ([] : List α) = [[]] :=
  rfl


theorem cyclicPermutations_cons (x : α) (l : List α) :
    cyclicPermutations (x :: l) = dropLast (zipWith (· ++ ·) (tails (x :: l)) (inits (x :: l))) :=
  rfl


theorem cyclicPermutations_of_ne_nil (l : List α) (h : l ≠ []) :
    cyclicPermutations l = dropLast (zipWith (· ++ ·) (tails l) (inits l)) := by
  /-
    α : Type u
    l : List α
    h : Ne l List.nil
    ⊢ Eq l.cyclicPermutations (List.zipWith (fun x1 x2 => HAppend.hAppend x1 x2) l …
  -/
  obtain ⟨hd, tl, rfl⟩ := exists_cons_of_ne_nil h
  /-
    case intro.intro
    α : Type u
    hd : α
    tl : List α
    h : Ne (List.cons hd tl) List.nil
    ⊢ Eq (List.cons hd tl).cyclicPermutations (List.zipWith (fun x1 x2 => HAppend. …
  -/
  exact cyclicPermutations_cons _ _
  /-
    🎉 no goals
  -/


theorem length_cyclicPermutations_cons (x : α) (l : List α) :
                                                              /-
                                                                α : Type u
                                                                x : α
                                                                l : List α
                                                                ⊢ Eq (List.cons x l).cyclicPermutations.length (HAdd.hAdd l.length 1)
                                                              -/
    length (cyclicPermutations (x :: l)) = length l + 1 := by simp [cyclicPermutations_cons]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
theorem length_cyclicPermutations_of_ne_nil (l : List α) (h : l ≠ []) :
                                                   /-
                                                     α : Type u
                                                     l : List α
                                                     h : Ne l List.nil
                                                     ⊢ Eq l.cyclicPermutations.length l.length
                                                   -/
    length (cyclicPermutations l) = length l := by simp [cyclicPermutations_of_ne_nil _ h]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
theorem cyclicPermutations_ne_nil : ∀ l : List α, cyclicPermutations l ≠ []
                  /-
                    α : Type u
                    a : α
                    l : List α
                    h : Eq (List.cons a l).cyclicPermutations List.nil
                    ⊢ False
                  -/
  | a::l, h => by simpa using congr_arg length h
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem getElem_cyclicPermutations (l : List α) (n : Nat) (h : n < length (cyclicPermutations l)) :
    (cyclicPermutations l)[n] = l.rotate n := by
  cases l with
  | nil => simp
  | cons a l =>
    simp only [cyclicPermutations_cons, getElem_dropLast, getElem_zipWith, getElem_tails,
      getElem_inits]
    rw [rotate_eq_drop_append_take (by simpa using h.le)]


theorem get_cyclicPermutations (l : List α) (n : Fin (length (cyclicPermutations l))) :
    (cyclicPermutations l).get n = l.rotate n := by
  /-
    α : Type u
    l : List α
    n : Fin l.cyclicPermutations.length
    ⊢ Eq (l.cyclicPermutations.get n) (l.rotate ↑n)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem head_cyclicPermutations (l : List α) :
    (cyclicPermutations l).head (cyclicPermutations_ne_nil l) = l := by
  /-
    α : Type u
    l : List α
    ⊢ Eq (l.cyclicPermutations.head ⋯) l
  -/
  have h : 0 < length (cyclicPermutations l) := length_pos_of_ne_nil (cyclicPermutations_ne_nil _)
  /-
    α : Type u
    l : List α
    h : LT.lt 0 l.cyclicPermutations.length
    ⊢ Eq (l.cyclicPermutations.head ⋯) l
  -/
  rw [← get_mk_zero h, get_cyclicPermutations, Fin.val_mk, rotate_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem head?_cyclicPermutations (l : List α) : (cyclicPermutations l).head? = l := by
  /-
    α : Type u
    l : List α
    ⊢ Eq l.cyclicPermutations.head? (Option.some l)
  -/
  rw [head?_eq_head, head_cyclicPermutations]
  /-
    🎉 no goals
  -/


theorem cyclicPermutations_injective : Function.Injective (@cyclicPermutations α) := fun l l' h ↦ by
  /-
    α : Type u
    l l' : List α
    h : Eq l.cyclicPermutations l'.cyclicPermutations
    ⊢ Eq l l'
  -/
  simpa using congr_arg head? h
  /-
    🎉 no goals
  -/


@[simp]
theorem cyclicPermutations_inj {l l' : List α} :
    cyclicPermutations l = cyclicPermutations l' ↔ l = l' :=
  cyclicPermutations_injective.eq_iff


theorem length_mem_cyclicPermutations (l : List α) (h : l' ∈ cyclicPermutations l) :
    length l' = length l := by
  /-
    α : Type u
    l' l : List α
    h : Membership.mem l.cyclicPermutations l'
    ⊢ Eq l'.length l.length
  -/
  obtain ⟨k, hk, rfl⟩ := get_of_mem h
  /-
    case intro.refl
    α : Type u
    l : List α
    k : Fin l.cyclicPermutations.length
    h : Membership.mem l.cyclicPermutations (l.cyclicPermutations.get k)
    ⊢ Eq (l.cyclicPermutations.get k).length l.length
  -/
  simp
  /-
    🎉 no goals
  -/


theorem mem_cyclicPermutations_self (l : List α) : l ∈ cyclicPermutations l := by
  /-
    α : Type u
    l : List α
    ⊢ Membership.mem l.cyclicPermutations l
  -/
  simpa using head_mem (cyclicPermutations_ne_nil l)
  /-
    🎉 no goals
  -/


@[simp]
theorem cyclicPermutations_rotate (l : List α) (k : ℕ) :
    (l.rotate k).cyclicPermutations = l.cyclicPermutations.rotate k := by
  have : (l.rotate k).cyclicPermutations.length = length (l.cyclicPermutations.rotate k) := by
    cases l
    · simp
    · rw [length_cyclicPermutations_of_ne_nil] <;> simp
  /-
    α : Type u
    l : List α
    k : Nat
    this : Eq (l.rotate k).cyclicPermutations.length (l.cyclicPermutations.rotate  …
    ⊢ Eq (l.rotate k).cyclicPermutations (l.cyclicPermutations.rotate k)
  -/
  refine ext_get this fun n hn hn' => ?_
  /-
    α : Type u
    l : List α
    k : Nat
    this : Eq (l.rotate k).cyclicPermutations.length (l.cyclicPermutations.rotate  …
    n : Nat
    hn : LT.lt n (l.rotate k).cyclicPermutations.length
    hn' : LT.lt n (l.cyclicPermutations.rotate k).length
    ⊢ Eq ((l.rotate k).cyclicPermutations.get ⟨n, hn⟩) ((l.cyclicPermutations.rota …
  -/
  rw [get_rotate, get_cyclicPermutations, rotate_rotate, ← rotate_mod, Nat.add_comm]
  /-
    α : Type u
    l : List α
    k : Nat
    this : Eq (l.rotate k).cyclicPermutations.length (l.cyclicPermutations.rotate  …
    n : Nat
    hn : LT.lt n (l.rotate k).cyclicPermutations.length
    hn' : LT.lt n (l.cyclicPermutations.rotate k).length
    ⊢ Eq (l.rotate (HMod.hMod (HAdd.hAdd (↑⟨n, hn⟩) k) l.length)) (l.cyclicPermuta …
  -/
              /-
                🎉 no goals
              -/
  cases l <;> simp
              /-
                🎉 no goals
              -/


@[simp]
theorem mem_cyclicPermutations_iff : l ∈ cyclicPermutations l' ↔ l ~r l' := by
  /-
    α : Type u
    l l' : List α
    ⊢ Iff (Membership.mem l'.cyclicPermutations l) (l.IsRotated l')
  -/
  constructor
    /-
      case mp
      α : Type u
      l l' : List α
      ⊢ Membership.mem l'.cyclicPermutations l → l.IsRotated l'
    -/
  · simp_rw [mem_iff_get, get_cyclicPermutations]
    /-
      case mp
      α : Type u
      l l' : List α
      ⊢ (Exists fun n => Eq (l'.rotate ↑n) l) → l.IsRotated l'
    -/
    rintro ⟨k, rfl⟩
    /-
      case mp.intro
      α : Type u
      l' : List α
      k : Fin l'.cyclicPermutations.length
      ⊢ (l'.rotate ↑k).IsRotated l'
    -/
    exact .forall _ _
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u
      l l' : List α
      ⊢ l.IsRotated l' → Membership.mem l'.cyclicPermutations l
    -/
  · rintro ⟨k, rfl⟩
    /-
      case mpr.intro
      α : Type u
      l : List α
      k : Nat
      ⊢ Membership.mem (l.rotate k).cyclicPermutations l
    -/
    rw [cyclicPermutations_rotate, mem_rotate]
    /-
      case mpr.intro
      α : Type u
      l : List α
      k : Nat
      ⊢ Membership.mem l.cyclicPermutations l
    -/
    apply mem_cyclicPermutations_self
    /-
      🎉 no goals
    -/


@[simp]
theorem cyclicPermutations_eq_nil_iff {l : List α} : cyclicPermutations l = [[]] ↔ l = [] :=
  cyclicPermutations_injective.eq_iff' rfl


@[simp]
theorem cyclicPermutations_eq_singleton_iff {l : List α} {x : α} :
    cyclicPermutations l = [[x]] ↔ l = [x] :=
  cyclicPermutations_injective.eq_iff' rfl


/-- If a `l : List α` is `Nodup l`, then all of its cyclic permutants are distinct. -/
protected theorem Nodup.cyclicPermutations {l : List α} (hn : Nodup l) :
    Nodup (cyclicPermutations l) := by
  /-
    α : Type u
    l : List α
    hn : l.Nodup
    ⊢ l.cyclicPermutations.Nodup
  -/
  rcases eq_or_ne l [] with rfl | hl
    /-
      case inl
      α : Type u
      hn : List.nil.Nodup
      ⊢ List.nil.cyclicPermutations.Nodup
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u
      l : List α
      hn : l.Nodup
      hl : Ne l List.nil
      ⊢ l.cyclicPermutations.Nodup
    -/
  · rw [nodup_iff_injective_get]
    /-
      case inr
      α : Type u
      l : List α
      hn : l.Nodup
      hl : Ne l List.nil
      ⊢ Function.Injective l.cyclicPermutations.get
    -/
    rintro ⟨i, hi⟩ ⟨j, hj⟩ h
    /-
      case inr.mk.mk
      α : Type u
      l : List α
      hn : l.Nodup
      hl : Ne l List.nil
      i : Nat
      hi : LT.lt i l.cyclicPermutations.length
      j : Nat
      hj : LT.lt j l.cyclicPermutations.length
      h : Eq (l.cyclicPermutations.get ⟨i, hi⟩) (l.cyclicPermutations.get ⟨j, hj⟩)
      ⊢ Eq ⟨i, hi⟩ ⟨j, hj⟩
    -/
    simp only [length_cyclicPermutations_of_ne_nil l hl] at hi hj
    /-
      case inr.mk.mk
      α : Type u
      l : List α
      hn : l.Nodup
      hl : Ne l List.nil
      i : Nat
      hi✝ : LT.lt i l.cyclicPermutations.length
      j : Nat
      hj✝ : LT.lt j l.cyclicPermutations.length
      h : Eq (l.cyclicPermutations.get ⟨i, hi✝⟩) (l.cyclicPermutations.get ⟨j, hj✝⟩)
      hi : LT.lt i l.length
      hj : LT.lt j l.length
      ⊢ Eq ⟨i, hi✝⟩ ⟨j, hj✝⟩
    -/
    simpa [hn.rotate_congr_iff, mod_eq_of_lt, *] using h
    /-
      🎉 no goals
    -/


protected theorem IsRotated.cyclicPermutations {l l' : List α} (h : l ~r l') :
    l.cyclicPermutations ~r l'.cyclicPermutations := by
  /-
    α : Type u
    l l' : List α
    h : l.IsRotated l'
    ⊢ l.cyclicPermutations.IsRotated l'.cyclicPermutations
  -/
  obtain ⟨k, rfl⟩ := h
  /-
    case intro
    α : Type u
    l : List α
    k : Nat
    ⊢ l.cyclicPermutations.IsRotated (l.rotate k).cyclicPermutations
  -/
  exact ⟨k, by simp⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem isRotated_cyclicPermutations_iff {l l' : List α} :
    l.cyclicPermutations ~r l'.cyclicPermutations ↔ l ~r l' := by
  /-
    α : Type u
    l l' : List α
    ⊢ Iff (l.cyclicPermutations.IsRotated l'.cyclicPermutations) (l.IsRotated l')
  -/
  simp only [IsRotated, ← cyclicPermutations_rotate, cyclicPermutations_inj]
  /-
    🎉 no goals
  -/


instance isRotatedDecidable (l l' : List α) : Decidable (l ~r l') :=
  decidable_of_iff' _ isRotated_iff_mem_map_range


instance {l l' : List α} : Decidable (IsRotated.setoid α l l') :=
  List.isRotatedDecidable _ _


