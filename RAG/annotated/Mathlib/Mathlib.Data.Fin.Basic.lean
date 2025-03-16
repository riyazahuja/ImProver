/-- Elimination principle for the empty set `Fin 0`, dependent version. -/
def finZeroElim {α : Fin 0 → Sort*} (x : Fin 0) : α x :=
  x.elim0


@[deprecated (since := "2024-02-15")] alias eq_of_veq := eq_of_val_eq

@[deprecated (since := "2024-02-15")] alias veq_of_eq := val_eq_of_eq

@[deprecated (since := "2024-08-13")] alias ne_of_vne := ne_of_val_ne

@[deprecated (since := "2024-08-13")] alias vne_of_ne := val_ne_of_ne


instance {n : ℕ} : CanLift ℕ (Fin n) Fin.val (· < n) where
  prf k hk := ⟨⟨k, hk⟩, rfl⟩


/-- A dependent variant of `Fin.elim0`. -/
def rec0 {α : Fin 0 → Sort*} (i : Fin 0) : α i := absurd i.2 (Nat.not_lt_zero _)


theorem val_injective : Function.Injective (@Fin.val n) :=
  @Fin.eq_of_val_eq n


/-- If you actually have an element of `Fin n`, then the `n` is always positive -/
lemma size_positive : Fin n → 0 < n := Fin.pos


lemma size_positive' [Nonempty (Fin n)] : 0 < n :=
  ‹Nonempty (Fin n)›.elim Fin.pos


protected theorem prop (a : Fin n) : a.val < n :=
  a.2


protected lemma lt_of_le_of_lt : a ≤ b → b < c → a < c := Nat.lt_of_le_of_lt

protected lemma lt_of_lt_of_le : a < b → b ≤ c → a < c := Nat.lt_of_lt_of_le

protected lemma le_rfl : a ≤ a := Nat.le_refl _

protected lemma lt_iff_le_and_ne : a < b ↔ a ≤ b ∧ a ≠ b := by
  /-
    n : Nat
    a b : Fin n
    ⊢ Iff (LT.lt a b) (And (LE.le a b) (Ne a b))
  -/
  rw [← val_ne_iff]; exact Nat.lt_iff_le_and_ne
                     /-
                       🎉 no goals
                     -/

protected lemma lt_or_lt_of_ne (h : a ≠ b) : a < b ∨ b < a := Nat.lt_or_lt_of_ne <| val_ne_iff.2 h

protected lemma lt_or_le (a b : Fin n) : a < b ∨ b ≤ a := Nat.lt_or_ge _ _

protected lemma le_or_lt (a b : Fin n) : a ≤ b ∨ b < a := (b.lt_or_le a).symm

protected lemma le_of_eq (hab : a = b) : a ≤ b := Nat.le_of_eq <| congr_arg val hab

protected lemma ge_of_eq (hab : a = b) : b ≤ a := Fin.le_of_eq hab.symm

protected lemma eq_or_lt_of_le : a ≤ b → a = b ∨ a < b := by
  /-
    n : Nat
    a b : Fin n
    ⊢ LE.le a b → Or (Eq a b) (LT.lt a b)
  -/
  rw [Fin.ext_iff]; exact Nat.eq_or_lt_of_le
                    /-
                      🎉 no goals
                    -/

protected lemma lt_or_eq_of_le : a ≤ b → a < b ∨ a = b := by
  /-
    n : Nat
    a b : Fin n
    ⊢ LE.le a b → Or (LT.lt a b) (Eq a b)
  -/
  rw [Fin.ext_iff]; exact Nat.lt_or_eq_of_le
                    /-
                      🎉 no goals
                    -/


lemma lt_last_iff_ne_last {a : Fin (n + 1)} : a < last n ↔ a ≠ last n := by
  /-
    n : Nat
    a : Fin (HAdd.hAdd n 1)
    ⊢ Iff (LT.lt a (Fin.last n)) (Ne a (Fin.last n))
  -/
  simp [Fin.lt_iff_le_and_ne, le_last]
  /-
    🎉 no goals
  -/


lemma ne_zero_of_lt {a b : Fin (n + 1)} (hab : a < b) : b ≠ 0 :=
  Fin.ne_of_gt <| Fin.lt_of_le_of_lt a.zero_le hab


lemma ne_last_of_lt {a b : Fin (n + 1)} (hab : a < b) : a ≠ last n :=
  Fin.ne_of_lt <| Fin.lt_of_lt_of_le hab b.le_last


/-- Equivalence between `Fin n` and `{ i // i < n }`. -/
@[simps apply symm_apply]
def equivSubtype : Fin n ≃ { i // i < n } where
  toFun a := ⟨a.1, a.2⟩
  invFun a := ⟨a.1, a.2⟩
  left_inv := fun ⟨_, _⟩ => rfl
  right_inv := fun ⟨_, _⟩ => rfl


theorem val_eq_val (a b : Fin n) : (a : ℕ) = b ↔ a = b :=
  Fin.ext_iff.symm


@[deprecated Fin.ext_iff (since := "2024-02-20")]
theorem eq_iff_veq (a b : Fin n) : a = b ↔ a.1 = b.1 :=
  Fin.ext_iff


theorem ne_iff_vne (a b : Fin n) : a ≠ b ↔ a.1 ≠ b.1 :=
  Fin.ext_iff.not

-- Porting note: I'm not sure if this comment still applies.
-- built-in reduction doesn't always work

@[simp, nolint simpNF]
theorem mk_eq_mk {a h a' h'} : @mk n a h = @mk n a' h' ↔ a = a' :=
  Fin.ext_iff

-- syntactic tautologies now


/-- Assume `k = l`. If two functions defined on `Fin k` and `Fin l` are equal on each element,
then they coincide (in the heq sense). -/
protected theorem heq_fun_iff {α : Sort*} {k l : ℕ} (h : k = l) {f : Fin k → α} {g : Fin l → α} :
    HEq f g ↔ ∀ i : Fin k, f i = g ⟨(i : ℕ), h ▸ i.2⟩ := by
  /-
    α : Sort u_1
    k l : Nat
    h : Eq k l
    f : Fin k → α
    g : Fin l → α
    ⊢ Iff (HEq f g) (∀ (i : Fin k), Eq (f i) (g ⟨↑i, ⋯⟩))
  -/
  subst h
  /-
    α : Sort u_1
    k : Nat
    f g : Fin k → α
    ⊢ Iff (HEq f g) (∀ (i : Fin k), Eq (f i) (g ⟨↑i, ⋯⟩))
  -/
  simp [funext_iff]
  /-
    🎉 no goals
  -/


/-- Assume `k = l` and `k' = l'`.
If two functions `Fin k → Fin k' → α` and `Fin l → Fin l' → α` are equal on each pair,
then they coincide (in the heq sense). -/
protected theorem heq_fun₂_iff {α : Sort*} {k l k' l' : ℕ} (h : k = l) (h' : k' = l')
    {f : Fin k → Fin k' → α} {g : Fin l → Fin l' → α} :
    HEq f g ↔ ∀ (i : Fin k) (j : Fin k'), f i j = g ⟨(i : ℕ), h ▸ i.2⟩ ⟨(j : ℕ), h' ▸ j.2⟩ := by
  /-
    α : Sort u_1
    k l k' l' : Nat
    h : Eq k l
    h' : Eq k' l'
    f : Fin k → Fin k' → α
    g : Fin l → Fin l' → α
    ⊢ Iff (HEq f g) (∀ (i : Fin k) (j : Fin k'), Eq (f i j) (g ⟨↑i, ⋯⟩ ⟨↑j, ⋯⟩))
  -/
  subst h
  /-
    α : Sort u_1
    k k' l' : Nat
    h' : Eq k' l'
    f : Fin k → Fin k' → α
    g : Fin k → Fin l' → α
    ⊢ Iff (HEq f g) (∀ (i : Fin k) (j : Fin k'), Eq (f i j) (g ⟨↑i, ⋯⟩ ⟨↑j, ⋯⟩))
  -/
  subst h'
  /-
    α : Sort u_1
    k k' : Nat
    f g : Fin k → Fin k' → α
    ⊢ Iff (HEq f g) (∀ (i : Fin k) (j : Fin k'), Eq (f i j) (g ⟨↑i, ⋯⟩ ⟨↑j, ⋯⟩))
  -/
  simp [funext_iff]
  /-
    🎉 no goals
  -/


/-- Two elements of `Fin k` and `Fin l` are heq iff their values in `ℕ` coincide. This requires
`k = l`. For the left implication without this assumption, see `val_eq_val_of_heq`. -/
protected theorem heq_ext_iff {k l : ℕ} (h : k = l) {i : Fin k} {j : Fin l} :
    HEq i j ↔ (i : ℕ) = (j : ℕ) := by
  /-
    k l : Nat
    h : Eq k l
    i : Fin k
    j : Fin l
    ⊢ Iff (HEq i j) (Eq ↑i ↑j)
  -/
  subst h
  /-
    k : Nat
    i j : Fin k
    ⊢ Iff (HEq i j) (Eq ↑i ↑j)
  -/
  simp [val_eq_val]
  /-
    🎉 no goals
  -/


theorem le_iff_val_le_val {a b : Fin n} : a ≤ b ↔ (a : ℕ) ≤ b :=
  Iff.rfl


/-- `a < b` as natural numbers if and only if `a < b` in `Fin n`. -/
@[norm_cast, simp]
theorem val_fin_lt {n : ℕ} {a b : Fin n} : (a : ℕ) < (b : ℕ) ↔ a < b :=
  Iff.rfl


/-- `a ≤ b` as natural numbers if and only if `a ≤ b` in `Fin n`. -/
@[norm_cast, simp]
theorem val_fin_le {n : ℕ} {a b : Fin n} : (a : ℕ) ≤ (b : ℕ) ↔ a ≤ b :=
  Iff.rfl


                                                      /-
                                                        n : Nat
                                                        a : Fin n
                                                        ⊢ Eq (Min.min (↑a) n) ↑a
                                                      -/
theorem min_val {a : Fin n} : min (a : ℕ) n = a := by simp
                                                      /-
                                                        🎉 no goals
                                                      -/


                                                      /-
                                                        n : Nat
                                                        a : Fin n
                                                        ⊢ Eq (Max.max (↑a) n) n
                                                      -/
theorem max_val {a : Fin n} : max (a : ℕ) n = n := by simp
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- The inclusion map `Fin n → ℕ` is an embedding. -/
@[simps apply]
def valEmbedding : Fin n ↪ ℕ :=
  ⟨val, val_injective⟩


@[simp]
theorem equivSubtype_symm_trans_valEmbedding :
    equivSubtype.symm.toEmbedding.trans valEmbedding = Embedding.subtype (· < n) :=
  rfl


/-- Use the ordering on `Fin n` for checking recursive definitions.

For example, the following definition is not accepted by the termination checker,
unless we declare the `WellFoundedRelation` instance:
```lean
def factorial {n : ℕ} : Fin n → ℕ
  | ⟨0, _⟩ := 1
  | ⟨i + 1, hi⟩ := (i + 1) * factorial ⟨i, i.lt_succ_self.trans hi⟩
```
-/
instance {n : ℕ} : WellFoundedRelation (Fin n) :=
  measure (val : Fin n → ℕ)


/-- Given a positive `n`, `Fin.ofNat' i` is `i % n` as an element of `Fin n`. -/
@[deprecated Fin.ofNat' (since := "2024-10-15")]
def ofNat'' [NeZero n] (i : ℕ) : Fin n :=
  ⟨i % n, mod_lt _ n.pos_of_neZero⟩
-- Porting note: `Fin.ofNat'` conflicts with something in core (there the hypothesis is `n > 0`),
-- so for now we make this double-prime `''`. This is also the reason for the dubious translation.


/--
The `Fin.val_zero` in `Lean` only applies in `Fin (n+1)`.
This one instead uses a `NeZero n` typeclass hypothesis.
-/
@[simp]
theorem val_zero' (n : ℕ) [NeZero n] : ((0 : Fin n) : ℕ) = 0 :=
  rfl


/--
The `Fin.zero_le` in `Lean` only applies in `Fin (n+1)`.
This one instead uses a `NeZero n` typeclass hypothesis.
-/
@[simp]
protected theorem zero_le' [NeZero n] (a : Fin n) : 0 ≤ a :=
  Nat.zero_le a.val


/--
The `Fin.pos_iff_ne_zero` in `Lean` only applies in `Fin (n+1)`.
This one instead uses a `NeZero n` typeclass hypothesis.
-/
theorem pos_iff_ne_zero' [NeZero n] (a : Fin n) : 0 < a ↔ a ≠ 0 := by
  /-
    n : Nat
    inst✝ : NeZero n
    a : Fin n
    ⊢ Iff (LT.lt 0 a) (Ne a 0)
  -/
  rw [← val_fin_lt, val_zero', Nat.pos_iff_ne_zero, Ne, Ne, Fin.ext_iff, val_zero']
  /-
    🎉 no goals
  -/


@[simp] lemma cast_eq_self (a : Fin n) : cast rfl a = a := rfl


@[simp] theorem cast_eq_zero {k l : ℕ} [NeZero k] [NeZero l]
                                                             /-
                                                               k l : Nat
                                                               inst✝¹ : NeZero k
                                                               inst✝ : NeZero l
                                                               h : Eq k l
                                                               x : Fin k
                                                               ⊢ Iff (Eq (Fin.cast h x) 0) (Eq x 0)
                                                             -/
    (h : k = l) (x : Fin k) : Fin.cast h x = 0 ↔ x = 0 := by simp [← val_eq_val]
                                                             /-
                                                               🎉 no goals
                                                             -/


lemma cast_injective {k l : ℕ} (h : k = l) : Injective (Fin.cast h) :=
                   /-
                     k l : Nat
                     h : Eq k l
                     a b : Fin k
                     hab : Eq (Fin.cast h a) (Fin.cast h b)
                     ⊢ Eq a b
                   -/
  fun a b hab ↦ by simpa [← val_eq_val] using hab
                   /-
                     🎉 no goals
                   -/


theorem rev_involutive : Involutive (rev : Fin n → Fin n) := rev_rev


/-- `Fin.rev` as an `Equiv.Perm`, the antitone involution `Fin n → Fin n` given by
`i ↦ n-(i+1)`. -/
@[simps! apply symm_apply]
def revPerm : Equiv.Perm (Fin n) :=
  Involutive.toPerm rev rev_involutive


theorem rev_injective : Injective (@rev n) :=
  rev_involutive.injective


theorem rev_surjective : Surjective (@rev n) :=
  rev_involutive.surjective


theorem rev_bijective : Bijective (@rev n) :=
  rev_involutive.bijective


@[simp]
theorem revPerm_symm : (@revPerm n).symm = revPerm :=
  rfl


theorem cast_rev (i : Fin n) (h : n = m) :
    cast h i.rev = (i.cast h).rev := by
  /-
    n m : Nat
    i : Fin n
    h : Eq n m
    ⊢ Eq (Fin.cast h i.rev) (Fin.cast h i).rev
  -/
  subst h; simp
           /-
             🎉 no goals
           -/


theorem rev_eq_iff {i j : Fin n} : rev i = j ↔ i = rev j := by
  /-
    n : Nat
    i j : Fin n
    ⊢ Iff (Eq i.rev j) (Eq i j.rev)
  -/
  rw [← rev_inj, rev_rev]
  /-
    🎉 no goals
  -/


theorem rev_ne_iff {i j : Fin n} : rev i ≠ j ↔ i ≠ rev j := rev_eq_iff.not


theorem rev_lt_iff {i j : Fin n} : rev i < j ↔ rev j < i := by
  /-
    n : Nat
    i j : Fin n
    ⊢ Iff (LT.lt i.rev j) (LT.lt j.rev i)
  -/
  rw [← rev_lt_rev, rev_rev]
  /-
    🎉 no goals
  -/


theorem rev_le_iff {i j : Fin n} : rev i ≤ j ↔ rev j ≤ i := by
  /-
    n : Nat
    i j : Fin n
    ⊢ Iff (LE.le i.rev j) (LE.le j.rev i)
  -/
  rw [← rev_le_rev, rev_rev]
  /-
    🎉 no goals
  -/


theorem lt_rev_iff {i j : Fin n} : i < rev j ↔ j < rev i := by
  /-
    n : Nat
    i j : Fin n
    ⊢ Iff (LT.lt i j.rev) (LT.lt j i.rev)
  -/
  rw [← rev_lt_rev, rev_rev]
  /-
    🎉 no goals
  -/


theorem le_rev_iff {i j : Fin n} : i ≤ rev j ↔ j ≤ rev i := by
  /-
    n : Nat
    i j : Fin n
    ⊢ Iff (LE.le i j.rev) (LE.le j i.rev)
  -/
  rw [← rev_le_rev, rev_rev]
  /-
    🎉 no goals
  -/

-- Porting note: this is now syntactically equal to `val_last`


@[simp] theorem val_rev_zero [NeZero n] : ((rev 0 : Fin n) : ℕ) = n.pred := rfl


theorem last_pos' [NeZero n] : 0 < last n := n.pos_of_neZero


theorem one_lt_last [NeZero n] : 1 < last (n + 1) := by
  /-
    n : Nat
    inst✝ : NeZero n
    ⊢ LT.lt 1 (Fin.last (HAdd.hAdd n 1))
  -/
  rw [lt_iff_val_lt_val, val_one, val_last, Nat.lt_add_left_iff_pos, Nat.pos_iff_ne_zero]
  /-
    n : Nat
    inst✝ : NeZero n
    ⊢ Ne n 0
  -/
  exact NeZero.ne n
  /-
    🎉 no goals
  -/


theorem coe_int_sub_eq_ite {n : Nat} (u v : Fin n) :
    ((u - v : Fin n) : Int) = if v ≤ u then (u - v : Int) else (u - v : Int) + n := by
  /-
    n : Nat
    u v : Fin n
    ⊢ Eq (↑↑(HSub.hSub u v)) (ite (LE.le v u) (HSub.hSub ↑↑u ↑↑v) (HAdd.hAdd (HSub …
  -/
  rw [Fin.sub_def]
  /-
    n : Nat
    u v : Fin n
    ⊢ Eq (↑↑⟨HMod.hMod (HAdd.hAdd (HSub.hSub n ↑v) ↑u) n, ⋯⟩) (ite (LE.le v u) (HS …
  -/
  split
    /-
      case isTrue
      n : Nat
      u v : Fin n
      h✝ : LE.le v u
      ⊢ Eq (↑↑⟨HMod.hMod (HAdd.hAdd (HSub.hSub n ↑v) ↑u) n, ⋯⟩) (HSub.hSub ↑↑u ↑↑v)
    -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
  · rw [ofNat_emod, Int.emod_eq_sub_self_emod, Int.emod_eq_of_lt] <;> omega
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
    /-
      case isFalse
      n : Nat
      u v : Fin n
      h✝ : Not (LE.le v u)
      ⊢ Eq (↑↑⟨HMod.hMod (HAdd.hAdd (HSub.hSub n ↑v) ↑u) n, ⋯⟩) (HAdd.hAdd (HSub.hSu …
    -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
  · rw [ofNat_emod, Int.emod_eq_of_lt] <;> omega
                                           /-
                                             🎉 no goals
                                           -/


theorem coe_int_sub_eq_mod {n : Nat} (u v : Fin n) :
    ((u - v : Fin n) : Int) = ((u : Int) - (v : Int)) % n := by
  /-
    n : Nat
    u v : Fin n
    ⊢ Eq (↑↑(HSub.hSub u v)) (HMod.hMod (HSub.hSub ↑↑u ↑↑v) ↑n)
  -/
  rw [coe_int_sub_eq_ite]
  /-
    n : Nat
    u v : Fin n
    ⊢ Eq (ite (LE.le v u) (HSub.hSub ↑↑u ↑↑v) (HAdd.hAdd (HSub.hSub ↑↑u ↑↑v) ↑n))  …
  -/
  split
    /-
      case isTrue
      n : Nat
      u v : Fin n
      h✝ : LE.le v u
      ⊢ Eq (HSub.hSub ↑↑u ↑↑v) (HMod.hMod (HSub.hSub ↑↑u ↑↑v) ↑n)
    -/
                               /-
                                 🎉 no goals
                               -/
  · rw [Int.emod_eq_of_lt] <;> omega
                               /-
                                 🎉 no goals
                               -/
    /-
      case isFalse
      n : Nat
      u v : Fin n
      h✝ : Not (LE.le v u)
      ⊢ Eq (HAdd.hAdd (HSub.hSub ↑↑u ↑↑v) ↑n) (HMod.hMod (HSub.hSub ↑↑u ↑↑v) ↑n)
    -/
                                                          /-
                                                            🎉 no goals
                                                          -/
  · rw [Int.emod_eq_add_self_emod, Int.emod_eq_of_lt] <;> omega
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem coe_int_add_eq_ite {n : Nat} (u v : Fin n) :
    ((u + v : Fin n) : Int) = if (u + v : ℕ) < n then (u + v : Int) else (u + v : Int) - n := by
  /-
    n : Nat
    u v : Fin n
    ⊢ Eq (↑↑(HAdd.hAdd u v)) (ite (LT.lt (HAdd.hAdd ↑u ↑v) n) (HAdd.hAdd ↑↑u ↑↑v)  …
  -/
  rw [Fin.add_def]
  /-
    n : Nat
    u v : Fin n
    ⊢ Eq (↑↑⟨HMod.hMod (HAdd.hAdd ↑u ↑v) n, ⋯⟩) (ite (LT.lt (HAdd.hAdd ↑u ↑v) n) ( …
  -/
  split
    /-
      case isTrue
      n : Nat
      u v : Fin n
      h✝ : LT.lt (HAdd.hAdd ↑u ↑v) n
      ⊢ Eq (↑↑⟨HMod.hMod (HAdd.hAdd ↑u ↑v) n, ⋯⟩) (HAdd.hAdd ↑↑u ↑↑v)
    -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
  · rw [ofNat_emod, Int.emod_eq_of_lt] <;> omega
                                           /-
                                             🎉 no goals
                                           -/
    /-
      case isFalse
      n : Nat
      u v : Fin n
      h✝ : Not (LT.lt (HAdd.hAdd ↑u ↑v) n)
      ⊢ Eq (↑↑⟨HMod.hMod (HAdd.hAdd ↑u ↑v) n, ⋯⟩) (HSub.hSub (HAdd.hAdd ↑↑u ↑↑v) ↑n)
    -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
  · rw [ofNat_emod, Int.emod_eq_sub_self_emod, Int.emod_eq_of_lt] <;> omega
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem coe_int_add_eq_mod {n : Nat} (u v : Fin n) :
    ((u + v : Fin n) : Int) = ((u : Int) + (v : Int)) % n := by
  /-
    n : Nat
    u v : Fin n
    ⊢ Eq (↑↑(HAdd.hAdd u v)) (HMod.hMod (HAdd.hAdd ↑↑u ↑↑v) ↑n)
  -/
  rw [coe_int_add_eq_ite]
  /-
    n : Nat
    u v : Fin n
    ⊢ Eq (ite (LT.lt (HAdd.hAdd ↑u ↑v) n) (HAdd.hAdd ↑↑u ↑↑v) (HSub.hSub (HAdd.hAd …
  -/
  split
    /-
      case isTrue
      n : Nat
      u v : Fin n
      h✝ : LT.lt (HAdd.hAdd ↑u ↑v) n
      ⊢ Eq (HAdd.hAdd ↑↑u ↑↑v) (HMod.hMod (HAdd.hAdd ↑↑u ↑↑v) ↑n)
    -/
                               /-
                                 🎉 no goals
                               -/
  · rw [Int.emod_eq_of_lt] <;> omega
                               /-
                                 🎉 no goals
                               -/
    /-
      case isFalse
      n : Nat
      u v : Fin n
      h✝ : Not (LT.lt (HAdd.hAdd ↑u ↑v) n)
      ⊢ Eq (HSub.hSub (HAdd.hAdd ↑↑u ↑↑v) ↑n) (HMod.hMod (HAdd.hAdd ↑↑u ↑↑v) ↑n)
    -/
                                                          /-
                                                            🎉 no goals
                                                          -/
  · rw [Int.emod_eq_sub_self_emod, Int.emod_eq_of_lt] <;> omega
                                                          /-
                                                            🎉 no goals
                                                          -/

-- Write `a + b` as `if (a + b : ℕ) < n then (a + b : ℤ) else (a + b : ℤ) - n` and
-- similarly `a - b` as `if (b : ℕ) ≤ a then (a - b : ℤ) else (a - b : ℤ) + n`.

attribute [fin_omega] Fin.lt_iff_val_lt_val Fin.le_iff_val_le_val

-- Rewrite `1 : Fin (n + 2)` to `1 : ℤ`

/--
Preprocessor for `omega` to handle inequalities in `Fin`.
Note that this involves a lot of case splitting, so may be slow.
-/
-- Further adjustment to the simp set can probably make this more powerful.
-- Please experiment and PR updates!
macro "fin_omega" : tactic => `(tactic|
  { try simp only [fin_omega, ← Int.ofNat_lt, ← Int.ofNat_le] at *
    omega })


@[simp]
theorem val_one' (n : ℕ) [NeZero n] : ((1 : Fin n) : ℕ) = 1 % n :=
  rfl

-- Porting note: Delete this lemma after porting

theorem val_one'' {n : ℕ} : ((1 : Fin (n + 1)) : ℕ) = 1 % (n + 1) :=
  rfl


instance nontrivial {n : ℕ} : Nontrivial (Fin (n + 2)) where
                                                    /-
                                                      n✝ m n : Nat
                                                      ⊢ Ne ↑0 ↑1
                                                    -/
  exists_pair_ne := ⟨0, 1, (ne_iff_vne 0 1).mpr (by simp [val_one, val_zero])⟩
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem nontrivial_iff_two_le : Nontrivial (Fin n) ↔ 2 ≤ n := by
  /-
    n : Nat
    ⊢ Iff (Nontrivial (Fin n)) (LE.le 2 n)
  -/
  rcases n with (_ | _ | n) <;>
  /-
    case zero
    ⊢ Iff (Nontrivial (Fin 0)) (LE.le 2 0)
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  simp [Fin.nontrivial, not_nontrivial, Nat.succ_le_iff]
  /-
    🎉 no goals
  -/


instance inhabitedFinOneAdd (n : ℕ) : Inhabited (Fin (1 + n)) :=
                               /-
                                 n✝ m n : Nat
                                 ⊢ NeZero (HAdd.hAdd 1 n)
                               -/
  haveI : NeZero (1 + n) := by rw [Nat.add_comm]; infer_instance
                                                  /-
                                                    🎉 no goals
                                                  -/
  inferInstance


@[simp]
theorem default_eq_zero (n : ℕ) [NeZero n] : (default : Fin n) = 0 :=
  rfl


instance instNatCast [NeZero n] : NatCast (Fin n) where
  natCast i := Fin.ofNat' n i


lemma natCast_def [NeZero n] (a : ℕ) : (a : Fin n) = ⟨a % n, mod_lt _ n.pos_of_neZero⟩ := rfl


theorem val_add_eq_ite {n : ℕ} (a b : Fin n) :
    (↑(a + b) : ℕ) = if n ≤ a + b then a + b - n else a + b := by
  rw [Fin.val_add, Nat.add_mod_eq_ite, Nat.mod_eq_of_lt (show ↑a < n from a.2),
    Nat.mod_eq_of_lt (show ↑b < n from b.2)]
--- Porting note: syntactically the same as the above


theorem val_add_eq_of_add_lt {n : ℕ} {a b : Fin n} (huv : a.val + b.val < n) :
    (a + b).val = a.val + b.val := by
  /-
    n : Nat
    a b : Fin n
    huv : LT.lt (HAdd.hAdd ↑a ↑b) n
    ⊢ Eq (↑(HAdd.hAdd a b)) (HAdd.hAdd ↑a ↑b)
  -/
  rw [val_add]
  /-
    n : Nat
    a b : Fin n
    huv : LT.lt (HAdd.hAdd ↑a ↑b) n
    ⊢ Eq (HMod.hMod (HAdd.hAdd ↑a ↑b) n) (HAdd.hAdd ↑a ↑b)
  -/
  simp [Nat.mod_eq_of_lt huv]
  /-
    🎉 no goals
  -/


lemma intCast_val_sub_eq_sub_add_ite {n : ℕ} (a b : Fin n) :
    ((a - b).val : ℤ) = a.val - b.val + if b ≤ a then 0 else n := by
  /-
    n : Nat
    a b : Fin n
    ⊢ Eq (↑↑(HSub.hSub a b)) (HAdd.hAdd (HSub.hSub ↑↑a ↑↑b) ↑(ite (LE.le b a) 0 n))
  -/
            /-
              🎉 no goals
            -/
  split <;> fin_omega
            /-
              🎉 no goals
            -/


@[simp]
theorem ofNat'_eq_cast (n : ℕ) [NeZero n] (a : ℕ) : Fin.ofNat' n a = a :=
  rfl


@[simp] lemma val_natCast (a n : ℕ) [NeZero n] : (a : Fin n).val = a % n := rfl


@[deprecated (since := "2024-04-17")]
alias val_nat_cast := val_natCast

-- Porting note: is this the right name for things involving `Nat.cast`?

/-- Converting an in-range number to `Fin (n + 1)` produces a result
whose value is the original number. -/
theorem val_cast_of_lt {n : ℕ} [NeZero n] {a : ℕ} (h : a < n) : (a : Fin n).val = a :=
  Nat.mod_eq_of_lt h


/-- If `n` is non-zero, converting the value of a `Fin n` to `Fin n` results
in the same value. -/
@[simp] theorem cast_val_eq_self {n : ℕ} [NeZero n] (a : Fin n) : (a.val : Fin n) = a :=
  Fin.ext <| val_cast_of_lt a.isLt

-- Porting note: this is syntactically the same as `val_cast_of_lt`

-- Porting note: this is syntactically the same as `cast_val_of_lt`

-- This is a special case of `CharP.cast_eq_zero` that doesn't require typeclass search

                                                                           /-
                                                                             n : Nat
                                                                             inst✝ : NeZero n
                                                                             ⊢ Eq (↑n) 0
                                                                           -/
@[simp high] lemma natCast_self (n : ℕ) [NeZero n] : (n : Fin n) = 0 := by ext; simp
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


@[deprecated (since := "2024-04-17")]
alias nat_cast_self := natCast_self


@[simp] lemma natCast_eq_zero {a n : ℕ} [NeZero n] : (a : Fin n) = 0 ↔ n ∣ a := by
  /-
    a n : Nat
    inst✝ : NeZero n
    ⊢ Iff (Eq (↑a) 0) (Dvd.dvd n a)
  -/
  simp [Fin.ext_iff, Nat.dvd_iff_mod_eq_zero]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias nat_cast_eq_zero := natCast_eq_zero


@[simp]
                                                                   /-
                                                                     n : Nat
                                                                     ⊢ Eq (↑n) (Fin.last n)
                                                                   -/
theorem natCast_eq_last (n) : (n : Fin (n + 1)) = Fin.last n := by ext; simp
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[deprecated (since := "2024-05-04")] alias cast_nat_eq_last := natCast_eq_last


theorem le_val_last (i : Fin (n + 1)) : i ≤ n := by
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    ⊢ LE.le i ↑n
  -/
  rw [Fin.natCast_eq_last]
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    ⊢ LE.le i (Fin.last n)
  -/
  exact Fin.le_last i
  /-
    🎉 no goals
  -/


lemma natCast_le_natCast (han : a ≤ n) (hbn : b ≤ n) : (a : Fin (n + 1)) ≤ b ↔ a ≤ b := by
  /-
    n a b : Nat
    han : LE.le a n
    hbn : LE.le b n
    ⊢ Iff (LE.le ↑a ↑b) (LE.le a b)
  -/
  rw [← Nat.lt_succ_iff] at han hbn
  /-
    n a b : Nat
    han : LT.lt a n.succ
    hbn : LT.lt b n.succ
    ⊢ Iff (LE.le ↑a ↑b) (LE.le a b)
  -/
  simp [le_iff_val_le_val, -val_fin_le, Nat.mod_eq_of_lt, han, hbn]
  /-
    🎉 no goals
  -/


lemma natCast_lt_natCast (han : a ≤ n) (hbn : b ≤ n) : (a : Fin (n + 1)) < b ↔ a < b := by
  /-
    n a b : Nat
    han : LE.le a n
    hbn : LE.le b n
    ⊢ Iff (LT.lt ↑a ↑b) (LT.lt a b)
  -/
  rw [← Nat.lt_succ_iff] at han hbn; simp [lt_iff_val_lt_val, Nat.mod_eq_of_lt, han, hbn]
                                     /-
                                       🎉 no goals
                                     -/


lemma natCast_mono (hbn : b ≤ n) (hab : a ≤ b) : (a : Fin (n + 1)) ≤ b :=
  (natCast_le_natCast (hab.trans hbn) hbn).2 hab


lemma natCast_strictMono (hbn : b ≤ n) (hab : a < b) : (a : Fin (n + 1)) < b :=
  (natCast_lt_natCast (hab.le.trans hbn) hbn).2 hab


                                                                       /-
                                                                         n : Nat
                                                                         a b : Fin n
                                                                         ⊢ Eq a.succ b.succ → Eq a b
                                                                       -/
lemma succ_injective (n : ℕ) : Injective (@Fin.succ n) := fun a b ↦ by simp [Fin.ext_iff]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


/-- `Fin.succ` as an `Embedding` -/
def succEmb (n : ℕ) : Fin n ↪ Fin (n + 1) where
  toFun := succ
  inj' := succ_injective _


@[simp]
theorem val_succEmb : ⇑(succEmb n) = Fin.succ := rfl


@[simp]
theorem exists_succ_eq {x : Fin (n + 1)} : (∃ y, Fin.succ y = x) ↔ x ≠ 0 :=
  ⟨fun ⟨_, hy⟩ => hy ▸ succ_ne_zero _, x.cases (fun h => h.irrefl.elim) (fun _ _ => ⟨_, rfl⟩)⟩


theorem exists_succ_eq_of_ne_zero {x : Fin (n + 1)} (h : x ≠ 0) :
    ∃ y, Fin.succ y = x := exists_succ_eq.mpr h


@[simp]
theorem succ_zero_eq_one' [NeZero n] : Fin.succ (0 : Fin n) = 1 := by
  /-
    n : Nat
    inst✝ : NeZero n
    ⊢ Eq (Fin.succ 0) 1
  -/
  cases n
    /-
      case zero
      inst✝ : NeZero 0
      ⊢ Eq (Fin.succ 0) 1
    -/
  · exact (NeZero.ne 0 rfl).elim
    /-
      🎉 no goals
    -/
    /-
      case succ
      n✝ : Nat
      inst✝ : NeZero (HAdd.hAdd n✝ 1)
      ⊢ Eq (Fin.succ 0) 1
    -/
  · rfl
    /-
      🎉 no goals
    -/


theorem one_pos' [NeZero n] : (0 : Fin (n + 1)) < 1 := succ_zero_eq_one' (n := n) ▸ succ_pos _

theorem zero_ne_one' [NeZero n] : (0 : Fin (n + 1)) ≠ 1 := Fin.ne_of_lt one_pos'


/--
The `Fin.succ_one_eq_two` in `Lean` only applies in `Fin (n+2)`.
This one instead uses a `NeZero n` typeclass hypothesis.
-/
@[simp]
theorem succ_one_eq_two' [NeZero n] : Fin.succ (1 : Fin (n + 1)) = 2 := by
  /-
    n : Nat
    inst✝ : NeZero n
    ⊢ Eq (Fin.succ 1) 2
  -/
  cases n
    /-
      case zero
      inst✝ : NeZero 0
      ⊢ Eq (Fin.succ 1) 2
    -/
  · exact (NeZero.ne 0 rfl).elim
    /-
      🎉 no goals
    -/
    /-
      case succ
      n✝ : Nat
      inst✝ : NeZero (HAdd.hAdd n✝ 1)
      ⊢ Eq (Fin.succ 1) 2
    -/
  · rfl
    /-
      🎉 no goals
    -/

-- Version of `succ_one_eq_two` to be used by `dsimp`.
-- Note the `'` swapped around due to a move to std4.


/--
The `Fin.le_zero_iff` in `Lean` only applies in `Fin (n+1)`.
This one instead uses a `NeZero n` typeclass hypothesis.
-/
@[simp]
theorem le_zero_iff' {n : ℕ} [NeZero n] {k : Fin n} : k ≤ 0 ↔ k = 0 :=
                          /-
                            n : Nat
                            inst✝ : NeZero n
                            k : Fin n
                            h : LE.le k 0
                            ⊢ Eq ↑k ↑0
                          -/
                                                         /-
                                                           🎉 no goals
                                                         -/
  ⟨fun h => Fin.ext <| by rw [Nat.eq_zero_of_le_zero h]; rfl, by rintro rfl; exact Nat.le_refl _⟩
                                                                             /-
                                                                               🎉 no goals
                                                                             -/

-- TODO: Move to Batteries

@[simp] lemma castLE_inj {hmn : m ≤ n} {a b : Fin m} : castLE hmn a = castLE hmn b ↔ a = b := by
  /-
    n m : Nat
    hmn : LE.le m n
    a b : Fin m
    ⊢ Iff (Eq (Fin.castLE hmn a) (Fin.castLE hmn b)) (Eq a b)
  -/
  simp [Fin.ext_iff]
  /-
    🎉 no goals
  -/


                                                                                  /-
                                                                                    n m : Nat
                                                                                    a b : Fin m
                                                                                    ⊢ Iff (Eq (Fin.castAdd n a) (Fin.castAdd n b)) (Eq a b)
                                                                                  -/
@[simp] lemma castAdd_inj {a b : Fin m} : castAdd n a = castAdd n b ↔ a = b := by simp [Fin.ext_iff]
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


attribute [simp] castSucc_inj


lemma castLE_injective (hmn : m ≤ n) : Injective (castLE hmn) :=
  fun _ _ hab ↦ Fin.ext (congr_arg val hab :)


lemma castAdd_injective (m n : ℕ) : Injective (@Fin.castAdd m n) := castLE_injective _


lemma castSucc_injective (n : ℕ) : Injective (@Fin.castSucc n) := castAdd_injective _ _


/-- `Fin.castLE` as an `Embedding`, `castLEEmb h i` embeds `i` into a larger `Fin` type. -/
@[simps! apply]
def castLEEmb (h : n ≤ m) : Fin n ↪ Fin m where
  toFun := castLE h
  inj' := castLE_injective _


@[simp, norm_cast] lemma coe_castLEEmb {m n} (hmn : m ≤ n) : castLEEmb hmn = castLE hmn := rfl

/- The next proof can be golfed a lot using `Fintype.card`.
It is written this way to define `ENat.card` and `Nat.card` without a `Fintype` dependency
(not done yet). -/

lemma nonempty_embedding_iff : Nonempty (Fin n ↪ Fin m) ↔ n ≤ m := by
  /-
    n m : Nat
    ⊢ Iff (Nonempty (Function.Embedding (Fin n) (Fin m))) (LE.le n m)
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ⟨castLEEmb h⟩⟩
  induction n generalizing m with
  | zero => exact m.zero_le
  | succ n ihn =>
    cases' h with e
    rcases exists_eq_succ_of_ne_zero (pos_iff_nonempty.2 (Nonempty.map e inferInstance)).ne'
      with ⟨m, rfl⟩
    refine Nat.succ_le_succ <| ihn ⟨?_⟩
    refine ⟨fun i ↦ (e.setValue 0 0 i.succ).pred (mt e.setValue_eq_iff.1 i.succ_ne_zero),
      fun i j h ↦ ?_⟩
    simpa only [pred_inj, EmbeddingLike.apply_eq_iff_eq, succ_inj] using h


lemma equiv_iff_eq : Nonempty (Fin m ≃ Fin n) ↔ m = n :=
  ⟨fun ⟨e⟩ ↦ le_antisymm (nonempty_embedding_iff.1 ⟨e⟩) (nonempty_embedding_iff.1 ⟨e.symm⟩),
    fun h ↦ h ▸ ⟨.refl _⟩⟩


@[simp] lemma castLE_castSucc {n m} (i : Fin n) (h : n + 1 ≤ m) :
    i.castSucc.castLE h = i.castLE (Nat.le_of_succ_le h) :=
  rfl


@[simp] lemma castLE_comp_castSucc {n m} (h : n + 1 ≤ m) :
    Fin.castLE h ∘ Fin.castSucc = Fin.castLE (Nat.le_of_succ_le h) :=
  rfl


@[simp] lemma castLE_rfl (n : ℕ) : Fin.castLE (le_refl n) = id :=
  rfl


@[simp]
theorem range_castLE {n k : ℕ} (h : n ≤ k) : Set.range (castLE h) = { i : Fin k | (i : ℕ) < n } :=
  Set.ext fun x => ⟨fun ⟨y, hy⟩ => hy ▸ y.2, fun hx => ⟨⟨x, hx⟩, rfl⟩⟩


@[simp]
theorem coe_of_injective_castLE_symm {n k : ℕ} (h : n ≤ k) (i : Fin k) (hi) :
    ((Equiv.ofInjective _ (castLE_injective h)).symm ⟨i, hi⟩ : ℕ) = i := by
  /-
    n k : Nat
    h : LE.le n k
    i : Fin k
    hi : Membership.mem (Set.range (Fin.castLE h)) i
    ⊢ Eq ↑((Equiv.ofInjective (Fin.castLE h) ⋯).symm ⟨i, hi⟩) ↑i
  -/
  rw [← coe_castLE h]
  /-
    n k : Nat
    h : LE.le n k
    i : Fin k
    hi : Membership.mem (Set.range (Fin.castLE h)) i
    ⊢ Eq ↑(Fin.castLE h ((Equiv.ofInjective (Fin.castLE h) ⋯).symm ⟨i, hi⟩)) ↑i
  -/
  exact congr_arg Fin.val (Equiv.apply_ofInjective_symm _ _)
  /-
    🎉 no goals
  -/


theorem leftInverse_cast (eq : n = m) : LeftInverse (cast eq.symm) (cast eq) :=
  fun _ => rfl


theorem rightInverse_cast (eq : n = m) : RightInverse (cast eq.symm) (cast eq) :=
  fun _ => rfl


theorem cast_lt_cast (eq : n = m) {a b : Fin n} : cast eq a < cast eq b ↔ a < b :=
  Iff.rfl


theorem cast_le_cast (eq : n = m) {a b : Fin n} : cast eq a ≤ cast eq b ↔ a ≤ b :=
  Iff.rfl


/-- The 'identity' equivalence between `Fin m` and `Fin n` when `m = n`. -/
@[simps]
def _root_.finCongr (eq : n = m) : Fin n ≃ Fin m where
  toFun := cast eq
  invFun := cast eq.symm
  left_inv := leftInverse_cast eq
  right_inv := rightInverse_cast eq


@[simp] lemma _root_.finCongr_apply_mk (h : m = n) (k : ℕ) (hk : k < m) :
    finCongr h ⟨k, hk⟩ = ⟨k, h ▸ hk⟩ := rfl


@[simp]
                                                                                      /-
                                                                                        n : Nat
                                                                                        h : optParam (Eq n n) ⋯
                                                                                        ⊢ Eq (finCongr h) (Equiv.refl (Fin n))
                                                                                      -/
lemma _root_.finCongr_refl (h : n = n := rfl) : finCongr h = Equiv.refl (Fin n) := by ext; simp
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/


@[simp] lemma _root_.finCongr_symm (h : m = n) : (finCongr h).symm = finCongr h.symm := rfl


@[simp] lemma _root_.finCongr_apply_coe (h : m = n) (k : Fin m) : (finCongr h k : ℕ) = k := rfl


lemma _root_.finCongr_symm_apply_coe (h : m = n) (k : Fin n) : ((finCongr h).symm k : ℕ) = k := rfl


/-- While in many cases `finCongr` is better than `Equiv.cast`/`cast`, sometimes we want to apply
a generic theorem about `cast`. -/
                                                                                    /-
                                                                                      n m : Nat
                                                                                      h : Eq n m
                                                                                      ⊢ Eq (finCongr h) (Equiv.cast ⋯)
                                                                                    -/
lemma _root_.finCongr_eq_equivCast (h : n = m) : finCongr h = .cast (h ▸ rfl) := by subst h; simp
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/


@[simp]
theorem cast_zero {n' : ℕ} [NeZero n] {h : n = n'} : cast h (0 : Fin n) =
       /-
         n m n' : Nat
         inst✝ : NeZero n
         h : Eq n n'
         ⊢ Fin n'
       -/
    by { haveI : NeZero n' := by {rw [← h]; infer_instance}; exact 0} := rfl
       /-
         🎉 no goals
       -/


/-- While in many cases `Fin.cast` is better than `Equiv.cast`/`cast`, sometimes we want to apply
a generic theorem about `cast`. -/
theorem cast_eq_cast (h : n = m) : (cast h : Fin n → Fin m) = _root_.cast (h ▸ rfl) := by
  /-
    n m : Nat
    h : Eq n m
    ⊢ Eq (Fin.cast h) (_root_.cast ⋯)
  -/
  subst h
  /-
    n : Nat
    ⊢ Eq (Fin.cast ⋯) (_root_.cast ⋯)
  -/
  ext
  /-
    case h.h
    n : Nat
    x✝ : Fin n
    ⊢ Eq ↑(Fin.cast ⋯ x✝) ↑(_root_.cast ⋯ x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- `Fin.castAdd` as an `Embedding`, `castAddEmb m i` embeds `i : Fin n` in `Fin (n+m)`.
See also `Fin.natAddEmb` and `Fin.addNatEmb`. -/
@[simps! apply]
def castAddEmb (m) : Fin n ↪ Fin (n + m) := castLEEmb (le_add_right n m)


/-- `Fin.castSucc` as an `Embedding`, `castSuccEmb i` embeds `i : Fin n` in `Fin (n+1)`. -/
@[simps! apply]
def castSuccEmb : Fin n ↪ Fin (n + 1) := castAddEmb _


@[simp, norm_cast] lemma coe_castSuccEmb : (castSuccEmb : Fin n → Fin (n + 1)) = Fin.castSucc := rfl


theorem castSucc_le_succ {n} (i : Fin n) : i.castSucc ≤ i.succ := Nat.le_succ i


@[simp] theorem castSucc_le_castSucc_iff {a b : Fin n} : castSucc a ≤ castSucc b ↔ a ≤ b := .rfl


@[simp] theorem succ_le_castSucc_iff {a b : Fin n} : succ a ≤ castSucc b ↔ a < b := by
  /-
    n : Nat
    a b : Fin n
    ⊢ Iff (LE.le a.succ b.castSucc) (LT.lt a b)
  -/
  rw [le_castSucc_iff, succ_lt_succ_iff]
  /-
    🎉 no goals
  -/


@[simp] theorem castSucc_lt_succ_iff {a b : Fin n} : castSucc a < succ b ↔ a ≤ b := by
  /-
    n : Nat
    a b : Fin n
    ⊢ Iff (LT.lt a.castSucc b.succ) (LE.le a b)
  -/
  rw [castSucc_lt_iff_succ_le, succ_le_succ_iff]
  /-
    🎉 no goals
  -/


theorem le_of_castSucc_lt_of_succ_lt {a b : Fin (n + 1)} {i : Fin n}
    (hl : castSucc i < a) (hu : b < succ i) : b < a := by
  /-
    n : Nat
    a b : Fin (HAdd.hAdd n 1)
    i : Fin n
    hl : LT.lt i.castSucc a
    hu : LT.lt b i.succ
    ⊢ LT.lt b a
  -/
  simp [Fin.lt_def, -val_fin_lt] at *; omega
                                       /-
                                         🎉 no goals
                                       -/


theorem castSucc_lt_or_lt_succ (p : Fin (n + 1)) (i : Fin n) : castSucc i < p ∨ p < i.succ := by
  /-
    n : Nat
    p : Fin (HAdd.hAdd n 1)
    i : Fin n
    ⊢ Or (LT.lt i.castSucc p) (LT.lt p i.succ)
  -/
  simp [Fin.lt_def, -val_fin_lt]; omega
                                  /-
                                    🎉 no goals
                                  -/


@[deprecated (since := "2024-05-30")] alias succAbove_lt_gt := castSucc_lt_or_lt_succ


theorem succ_le_or_le_castSucc (p : Fin (n + 1)) (i : Fin n) : succ i ≤ p ∨ p ≤ i.castSucc := by
  /-
    n : Nat
    p : Fin (HAdd.hAdd n 1)
    i : Fin n
    ⊢ Or (LE.le i.succ p) (LE.le p i.castSucc)
  -/
  rw [le_castSucc_iff, ← castSucc_lt_iff_succ_le]
  /-
    n : Nat
    p : Fin (HAdd.hAdd n 1)
    i : Fin n
    ⊢ Or (LT.lt i.castSucc p) (LT.lt p i.succ)
  -/
  exact p.castSucc_lt_or_lt_succ i
  /-
    🎉 no goals
  -/


theorem exists_castSucc_eq_of_ne_last {x : Fin (n + 1)} (h : x ≠ (last _)) :
    ∃ y, Fin.castSucc y = x := exists_castSucc_eq.mpr h


theorem forall_fin_succ' {P : Fin (n + 1) → Prop} :
    (∀ i, P i) ↔ (∀ i : Fin n, P i.castSucc) ∧ P (.last _) :=
  ⟨fun H => ⟨fun _ => H _, H _⟩, fun ⟨H0, H1⟩ i => Fin.lastCases H1 H0 i⟩

-- to match `Fin.eq_zero_or_eq_succ`

theorem eq_castSucc_or_eq_last {n : Nat} (i : Fin (n + 1)) :
    (∃ j : Fin n, i = j.castSucc) ∨ i = last n := i.lastCases (Or.inr rfl) (Or.inl ⟨·, rfl⟩)


theorem exists_fin_succ' {P : Fin (n + 1) → Prop} :
    (∃ i, P i) ↔ (∃ i : Fin n, P i.castSucc) ∨ P (.last _) :=
  ⟨fun ⟨i, h⟩ => Fin.lastCases Or.inr (fun i hi => Or.inl ⟨i, hi⟩) i h,
   fun h => h.elim (fun ⟨i, hi⟩ => ⟨i.castSucc, hi⟩) (fun h => ⟨.last _, h⟩)⟩


/--
The `Fin.castSucc_zero` in `Lean` only applies in `Fin (n+1)`.
This one instead uses a `NeZero n` typeclass hypothesis.
-/
@[simp]
theorem castSucc_zero' [NeZero n] : castSucc (0 : Fin n) = 0 := rfl


/-- `castSucc i` is positive when `i` is positive.

The `Fin.castSucc_pos` in `Lean` only applies in `Fin (n+1)`.
This one instead uses a `NeZero n` typeclass hypothesis. -/
theorem castSucc_pos' [NeZero n] {i : Fin n} (h : 0 < i) : 0 < castSucc i := by
  /-
    n : Nat
    inst✝ : NeZero n
    i : Fin n
    h : LT.lt 0 i
    ⊢ LT.lt 0 i.castSucc
  -/
  simpa [lt_iff_val_lt_val] using h
  /-
    🎉 no goals
  -/


/--
The `Fin.castSucc_eq_zero_iff` in `Lean` only applies in `Fin (n+1)`.
This one instead uses a `NeZero n` typeclass hypothesis.
-/
@[simp]
theorem castSucc_eq_zero_iff' [NeZero n] (a : Fin n) : castSucc a = 0 ↔ a = 0 :=
                                                /-
                                                  n : Nat
                                                  inst✝ : NeZero n
                                                  a : Fin n
                                                  ⊢ Iff (Eq ↑a ↑0) (Eq ↑a.castSucc ↑0)
                                                -/
  Fin.ext_iff.trans <| (Fin.ext_iff.trans <| by simp).symm
                                                /-
                                                  🎉 no goals
                                                -/


/--
The `Fin.castSucc_ne_zero_iff` in `Lean` only applies in `Fin (n+1)`.
This one instead uses a `NeZero n` typeclass hypothesis.
-/
theorem castSucc_ne_zero_iff' [NeZero n] (a : Fin n) : castSucc a ≠ 0 ↔ a ≠ 0 :=
  not_iff_not.mpr <| castSucc_eq_zero_iff' a


theorem castSucc_ne_zero_of_lt {p i : Fin n} (h : p < i) : castSucc i ≠ 0 := by
  /-
    n : Nat
    p i : Fin n
    h : LT.lt p i
    ⊢ Ne i.castSucc 0
  -/
  cases n
    /-
      case zero
      p i : Fin 0
      h : LT.lt p i
      ⊢ Ne i.castSucc 0
    -/
  · exact i.elim0
    /-
      🎉 no goals
    -/
    /-
      case succ
      n✝ : Nat
      p i : Fin (HAdd.hAdd n✝ 1)
      h : LT.lt p i
      ⊢ Ne i.castSucc 0
    -/
  · rw [castSucc_ne_zero_iff', Ne, Fin.ext_iff]
    /-
      case succ
      n✝ : Nat
      p i : Fin (HAdd.hAdd n✝ 1)
      h : LT.lt p i
      ⊢ Not (Eq ↑i ↑0)
    -/
    exact ((zero_le _).trans_lt h).ne'
    /-
      🎉 no goals
    -/


theorem succ_ne_last_iff (a : Fin (n + 1)) : succ a ≠ last (n + 1) ↔ a ≠ last n :=
  not_iff_not.mpr <| succ_eq_last_succ


theorem succ_ne_last_of_lt {p i : Fin n} (h : i < p) : succ i ≠ last n := by
  /-
    n : Nat
    p i : Fin n
    h : LT.lt i p
    ⊢ Ne i.succ (Fin.last n)
  -/
  cases n
    /-
      case zero
      p i : Fin 0
      h : LT.lt i p
      ⊢ Ne i.succ (Fin.last 0)
    -/
  · exact i.elim0
    /-
      🎉 no goals
    -/
    /-
      case succ
      n✝ : Nat
      p i : Fin (HAdd.hAdd n✝ 1)
      h : LT.lt i p
      ⊢ Ne i.succ (Fin.last (HAdd.hAdd n✝ 1))
    -/
  · rw [succ_ne_last_iff, Ne, Fin.ext_iff]
    /-
      case succ
      n✝ : Nat
      p i : Fin (HAdd.hAdd n✝ 1)
      h : LT.lt i p
      ⊢ Not (Eq ↑i ↑(Fin.last n✝))
    -/
    exact ((le_last _).trans_lt' h).ne
    /-
      🎉 no goals
    -/


@[norm_cast, simp]
theorem coe_eq_castSucc {a : Fin n} : (a : Fin (n + 1)) = castSucc a := by
  /-
    n : Nat
    a : Fin n
    ⊢ Eq (↑↑a) a.castSucc
  -/
  ext
  /-
    case h
    n : Nat
    a : Fin n
    ⊢ Eq ↑↑↑a ↑a.castSucc
  -/
  exact val_cast_of_lt (Nat.lt.step a.is_lt)
  /-
    🎉 no goals
  -/


theorem coe_succ_lt_iff_lt {n : ℕ} {j k : Fin n} : (j : Fin <| n + 1) < k ↔ j < k := by
  /-
    n : Nat
    j k : Fin n
    ⊢ Iff (LT.lt ↑↑j ↑↑k) (LT.lt j k)
  -/
  simp only [coe_eq_castSucc, castSucc_lt_castSucc_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem range_castSucc {n : ℕ} : Set.range (castSucc : Fin n → Fin n.succ) =
                                                                 /-
                                                                   n : Nat
                                                                   ⊢ LE.le n (HAdd.hAdd n 1)
                                                                 -/
    ({ i | (i : ℕ) < n } : Set (Fin n.succ)) := range_castLE (by omega)
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp]
theorem coe_of_injective_castSucc_symm {n : ℕ} (i : Fin n.succ) (hi) :
    ((Equiv.ofInjective castSucc (castSucc_injective _)).symm ⟨i, hi⟩ : ℕ) = i := by
  /-
    n : Nat
    i : Fin n.succ
    hi : Membership.mem (Set.range Fin.castSucc) i
    ⊢ Eq ↑((Equiv.ofInjective Fin.castSucc ⋯).symm ⟨i, hi⟩) ↑i
  -/
  rw [← coe_castSucc]
  /-
    n : Nat
    i : Fin n.succ
    hi : Membership.mem (Set.range Fin.castSucc) i
    ⊢ Eq ↑((Equiv.ofInjective Fin.castSucc ⋯).symm ⟨i, hi⟩).castSucc ↑i
  -/
  exact congr_arg val (Equiv.apply_ofInjective_symm _ _)
  /-
    🎉 no goals
  -/


/-- `Fin.addNat` as an `Embedding`, `addNatEmb m i` adds `m` to `i`, generalizes `Fin.succ`. -/
@[simps! apply]
def addNatEmb (m) : Fin n ↪ Fin (n + m) where
  toFun := (addNat · m)
                 /-
                   n m✝ m : Nat
                   a b : Fin n
                   ⊢ Eq ((fun x => x.addNat m) a) ((fun x => x.addNat m) b) → Eq a b
                 -/
  inj' a b := by simp [Fin.ext_iff]
                 /-
                   🎉 no goals
                 -/


/-- `Fin.natAdd` as an `Embedding`, `natAddEmb n i` adds `n` to `i` "on the left". -/
@[simps! apply]
def natAddEmb (n) {m} : Fin m ↪ Fin (n + m) where
  toFun := natAdd n
                 /-
                   n✝ m✝ n m : Nat
                   a b : Fin m
                   ⊢ Eq (Fin.natAdd n a) (Fin.natAdd n b) → Eq a b
                 -/
  inj' a b := by simp [Fin.ext_iff]
                 /-
                   🎉 no goals
                 -/


theorem pred_one' [NeZero n] (h := (zero_ne_one' (n := n)).symm) :
    Fin.pred (1 : Fin (n + 1)) h = 0 := by
  /-
    n : Nat
    inst✝ : NeZero n
    h : optParam (Ne 1 0) ⋯
    ⊢ Eq (Fin.pred 1 h) 0
  -/
  simp_rw [Fin.ext_iff, coe_pred, val_one', val_zero', Nat.sub_eq_zero_iff_le, Nat.mod_le]
  /-
    🎉 no goals
  -/


theorem pred_last (h := Fin.ext_iff.not.2 last_pos'.ne') :
                                         /-
                                           n : Nat
                                           h : optParam (Not (Eq (Fin.last (HAdd.hAdd n 1)) 0)) ⋯
                                           ⊢ Eq ((Fin.last (HAdd.hAdd n 1)).pred h) (Fin.last n)
                                         -/
    pred (last (n + 1)) h = last n := by simp_rw [← succ_last, pred_succ]
                                         /-
                                           🎉 no goals
                                         -/


theorem pred_lt_iff {j : Fin n} {i : Fin (n + 1)} (hi : i ≠ 0) : pred i hi < j ↔ i < succ j := by
  /-
    n : Nat
    j : Fin n
    i : Fin (HAdd.hAdd n 1)
    hi : Ne i 0
    ⊢ Iff (LT.lt (i.pred hi) j) (LT.lt i j.succ)
  -/
  rw [← succ_lt_succ_iff, succ_pred]
  /-
    🎉 no goals
  -/

theorem lt_pred_iff {j : Fin n} {i : Fin (n + 1)} (hi : i ≠ 0) : j < pred i hi ↔ succ j < i := by
  /-
    n : Nat
    j : Fin n
    i : Fin (HAdd.hAdd n 1)
    hi : Ne i 0
    ⊢ Iff (LT.lt j (i.pred hi)) (LT.lt j.succ i)
  -/
  rw [← succ_lt_succ_iff, succ_pred]
  /-
    🎉 no goals
  -/

theorem pred_le_iff {j : Fin n} {i : Fin (n + 1)} (hi : i ≠ 0) : pred i hi ≤ j ↔ i ≤ succ j := by
  /-
    n : Nat
    j : Fin n
    i : Fin (HAdd.hAdd n 1)
    hi : Ne i 0
    ⊢ Iff (LE.le (i.pred hi) j) (LE.le i j.succ)
  -/
  rw [← succ_le_succ_iff, succ_pred]
  /-
    🎉 no goals
  -/

theorem le_pred_iff {j : Fin n} {i : Fin (n + 1)} (hi : i ≠ 0) : j ≤ pred i hi ↔ succ j ≤ i := by
  /-
    n : Nat
    j : Fin n
    i : Fin (HAdd.hAdd n 1)
    hi : Ne i 0
    ⊢ Iff (LE.le j (i.pred hi)) (LE.le j.succ i)
  -/
  rw [← succ_le_succ_iff, succ_pred]
  /-
    🎉 no goals
  -/


theorem castSucc_pred_eq_pred_castSucc {a : Fin (n + 1)} (ha : a ≠ 0)
    (ha' := castSucc_ne_zero_iff.mpr ha) :
    (a.pred ha).castSucc = (castSucc a).pred ha' := rfl


theorem castSucc_pred_add_one_eq {a : Fin (n + 1)} (ha : a ≠ 0) :
    (a.pred ha).castSucc + 1 = a := by
  /-
    n : Nat
    a : Fin (HAdd.hAdd n 1)
    ha : Ne a 0
    ⊢ Eq (HAdd.hAdd (a.pred ha).castSucc 1) a
  -/
  cases' a using cases with a
    /-
      case zero
      n : Nat
      ha : Ne 0 0
      ⊢ Eq (HAdd.hAdd (Fin.pred 0 ha).castSucc 1) 0
    -/
  · exact (ha rfl).elim
    /-
      🎉 no goals
    -/
    /-
      case succ
      n : Nat
      a : Fin n
      ha : Ne a.succ 0
      ⊢ Eq (HAdd.hAdd (a.succ.pred ha).castSucc 1) a.succ
    -/
  · rw [pred_succ, coeSucc_eq_succ]
    /-
      🎉 no goals
    -/


theorem le_pred_castSucc_iff {a b : Fin (n + 1)} (ha : castSucc a ≠ 0) :
    b ≤ (castSucc a).pred ha ↔ b < a := by
  /-
    n : Nat
    a b : Fin (HAdd.hAdd n 1)
    ha : Ne a.castSucc 0
    ⊢ Iff (LE.le b (a.castSucc.pred ha)) (LT.lt b a)
  -/
  rw [le_pred_iff, succ_le_castSucc_iff]
  /-
    🎉 no goals
  -/


theorem pred_castSucc_lt_iff {a b : Fin (n + 1)} (ha : castSucc a ≠ 0) :
    (castSucc a).pred ha < b ↔ a ≤ b := by
  /-
    n : Nat
    a b : Fin (HAdd.hAdd n 1)
    ha : Ne a.castSucc 0
    ⊢ Iff (LT.lt (a.castSucc.pred ha) b) (LE.le a b)
  -/
  rw [pred_lt_iff, castSucc_lt_succ_iff]
  /-
    🎉 no goals
  -/


theorem pred_castSucc_lt {a : Fin (n + 1)} (ha : castSucc a ≠ 0) :
                                   /-
                                     n : Nat
                                     a : Fin (HAdd.hAdd n 1)
                                     ha : Ne a.castSucc 0
                                     ⊢ LT.lt (a.castSucc.pred ha) a
                                   -/
    (castSucc a).pred ha < a := by rw [pred_castSucc_lt_iff, le_def]
                                   /-
                                     🎉 no goals
                                   -/


theorem le_castSucc_pred_iff {a b : Fin (n + 1)} (ha : a ≠ 0) :
    b ≤ castSucc (a.pred ha) ↔ b < a := by
  /-
    n : Nat
    a b : Fin (HAdd.hAdd n 1)
    ha : Ne a 0
    ⊢ Iff (LE.le b (a.pred ha).castSucc) (LT.lt b a)
  -/
  rw [castSucc_pred_eq_pred_castSucc, le_pred_castSucc_iff]
  /-
    🎉 no goals
  -/


theorem castSucc_pred_lt_iff {a b : Fin (n + 1)} (ha : a ≠ 0) :
    castSucc (a.pred ha) < b ↔ a ≤ b := by
  /-
    n : Nat
    a b : Fin (HAdd.hAdd n 1)
    ha : Ne a 0
    ⊢ Iff (LT.lt (a.pred ha).castSucc b) (LE.le a b)
  -/
  rw [castSucc_pred_eq_pred_castSucc, pred_castSucc_lt_iff]
  /-
    🎉 no goals
  -/


theorem castSucc_pred_lt {a : Fin (n + 1)} (ha : a ≠ 0) :
                                   /-
                                     n : Nat
                                     a : Fin (HAdd.hAdd n 1)
                                     ha : Ne a 0
                                     ⊢ LT.lt (a.pred ha).castSucc a
                                   -/
    castSucc (a.pred ha) < a := by rw [castSucc_pred_lt_iff, le_def]
                                   /-
                                     🎉 no goals
                                   -/


/-- `castPred i` sends `i : Fin (n + 1)` to `Fin n` as long as i ≠ last n. -/
@[inline] def castPred (i : Fin (n + 1)) (h : i ≠ last n) : Fin n := castLT i (val_lt_last h)


@[simp]
lemma castLT_eq_castPred (i : Fin (n + 1)) (h : i < last _) (h' := Fin.ext_iff.not.2 h.ne) :
    castLT i h = castPred i h' := rfl


@[simp]
lemma coe_castPred (i : Fin (n + 1)) (h : i ≠ last _) : (castPred i h : ℕ) = i := rfl


@[simp]
theorem castPred_castSucc {i : Fin n} (h' := Fin.ext_iff.not.2 (castSucc_lt_last i).ne) :
    castPred (castSucc i) h' = i := rfl


@[simp]
theorem castSucc_castPred (i : Fin (n + 1)) (h : i ≠ last n) :
    castSucc (i.castPred h) = i := by
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    h : Ne i (Fin.last n)
    ⊢ Eq (i.castPred h).castSucc i
  -/
  rcases exists_castSucc_eq.mpr h with ⟨y, rfl⟩
  /-
    case intro
    n : Nat
    y : Fin n
    h : Ne y.castSucc (Fin.last n)
    ⊢ Eq (y.castSucc.castPred h).castSucc y.castSucc
  -/
  rw [castPred_castSucc]
  /-
    🎉 no goals
  -/


theorem castPred_eq_iff_eq_castSucc (i : Fin (n + 1)) (hi : i ≠ last _) (j : Fin n) :
    castPred i hi = j ↔ i = castSucc j :=
               /-
                 n : Nat
                 i : Fin (HAdd.hAdd n 1)
                 hi : Ne i (Fin.last n)
                 j : Fin n
                 h : Eq (i.castPred hi) j
                 ⊢ Eq i j.castSucc
               -/
               /-
                 🎉 no goals
               -/
  ⟨fun h => by rw [← h, castSucc_castPred], fun h => by simp_rw [h, castPred_castSucc]⟩
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
theorem castPred_mk (i : ℕ) (h₁ : i < n) (h₂ := h₁.trans (Nat.lt_succ_self _))
    (h₃ : ⟨i, h₂⟩ ≠ last _ := (ne_iff_vne _ _).mpr (val_last _ ▸ h₁.ne)) :
  castPred ⟨i, h₂⟩ h₃ = ⟨i, h₁⟩ := rfl


theorem castPred_le_castPred_iff {i j : Fin (n + 1)} {hi : i ≠ last n} {hj : j ≠ last n} :
    castPred i hi ≤ castPred j hj ↔ i ≤ j := Iff.rfl


theorem castPred_lt_castPred_iff {i j : Fin (n + 1)} {hi : i ≠ last n} {hj : j ≠ last n} :
    castPred i hi < castPred j hj ↔ i < j := Iff.rfl


theorem castPred_lt_iff {j : Fin n} {i : Fin (n + 1)} (hi : i ≠ last n) :
    castPred i hi < j ↔ i < castSucc j := by
  /-
    n : Nat
    j : Fin n
    i : Fin (HAdd.hAdd n 1)
    hi : Ne i (Fin.last n)
    ⊢ Iff (LT.lt (i.castPred hi) j) (LT.lt i j.castSucc)
  -/
  rw [← castSucc_lt_castSucc_iff, castSucc_castPred]
  /-
    🎉 no goals
  -/


theorem lt_castPred_iff {j : Fin n} {i : Fin (n + 1)} (hi : i ≠ last n) :
    j < castPred i hi ↔ castSucc j < i := by
  /-
    n : Nat
    j : Fin n
    i : Fin (HAdd.hAdd n 1)
    hi : Ne i (Fin.last n)
    ⊢ Iff (LT.lt j (i.castPred hi)) (LT.lt j.castSucc i)
  -/
  rw [← castSucc_lt_castSucc_iff, castSucc_castPred]
  /-
    🎉 no goals
  -/


theorem castPred_le_iff {j : Fin n} {i : Fin (n + 1)} (hi : i ≠ last n) :
    castPred i hi ≤ j ↔ i ≤ castSucc j := by
  /-
    n : Nat
    j : Fin n
    i : Fin (HAdd.hAdd n 1)
    hi : Ne i (Fin.last n)
    ⊢ Iff (LE.le (i.castPred hi) j) (LE.le i j.castSucc)
  -/
  rw [← castSucc_le_castSucc_iff, castSucc_castPred]
  /-
    🎉 no goals
  -/


theorem le_castPred_iff {j : Fin n} {i : Fin (n + 1)} (hi : i ≠ last n) :
    j ≤ castPred i hi ↔ castSucc j ≤ i := by
  /-
    n : Nat
    j : Fin n
    i : Fin (HAdd.hAdd n 1)
    hi : Ne i (Fin.last n)
    ⊢ Iff (LE.le j (i.castPred hi)) (LE.le j.castSucc i)
  -/
  rw [← castSucc_le_castSucc_iff, castSucc_castPred]
  /-
    🎉 no goals
  -/


theorem castPred_inj {i j : Fin (n + 1)} {hi : i ≠ last n} {hj : j ≠ last n} :
    castPred i hi = castPred j hj ↔ i = j := by
  /-
    n : Nat
    i j : Fin (HAdd.hAdd n 1)
    hi : Ne i (Fin.last n)
    hj : Ne j (Fin.last n)
    ⊢ Iff (Eq (i.castPred hi) (j.castPred hj)) (Eq i j)
  -/
  simp_rw [Fin.ext_iff, le_antisymm_iff, ← le_def, castPred_le_castPred_iff]
  /-
    🎉 no goals
  -/


theorem castPred_zero' [NeZero n] (h := Fin.ext_iff.not.2 last_pos'.ne) :
    castPred (0 : Fin (n + 1)) h = 0 := rfl


theorem castPred_zero (h := Fin.ext_iff.not.2 last_pos.ne)  :
    castPred (0 : Fin (n + 2)) h = 0 := rfl


@[simp]
theorem castPred_one [NeZero n] (h := Fin.ext_iff.not.2 one_lt_last.ne) :
    castPred (1 : Fin (n + 2)) h = 1 := by
  /-
    n : Nat
    inst✝ : NeZero n
    h : optParam (Not (Eq 1 (Fin.last (HAdd.hAdd n 1)))) ⋯
    ⊢ Eq (Fin.castPred 1 h) 1
  -/
  cases n
    /-
      case zero
      inst✝ : NeZero 0
      h : optParam (Not (Eq 1 (Fin.last (HAdd.hAdd 0 1)))) ⋯
      ⊢ Eq (Fin.castPred 1 h) 1
    -/
  · exact subsingleton_one.elim _ 1
    /-
      🎉 no goals
    -/
    /-
      case succ
      n✝ : Nat
      inst✝ : NeZero (HAdd.hAdd n✝ 1)
      h : optParam (Not (Eq 1 (Fin.last (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)))) ⋯
      ⊢ Eq (Fin.castPred 1 h) 1
    -/
  · rfl
    /-
      🎉 no goals
    -/


theorem rev_pred {i : Fin (n + 1)} (h : i ≠ 0) (h' := rev_ne_iff.mpr ((rev_last _).symm ▸ h)) :
    rev (pred i h) = castPred (rev i) h' := by
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    h : Ne i 0
    h' : optParam (Ne i.rev (Fin.last n)) ⋯
    ⊢ Eq (i.pred h).rev (i.rev.castPred h')
  -/
  rw [← castSucc_inj, castSucc_castPred, ← rev_succ, succ_pred]
  /-
    🎉 no goals
  -/


theorem rev_castPred {i : Fin (n + 1)}
    (h : i ≠ last n) (h' := rev_ne_iff.mpr ((rev_zero _).symm ▸ h)) :
    rev (castPred i h) = pred (rev i) h' := by
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    h : Ne i (Fin.last n)
    h' : optParam (Ne i.rev 0) ⋯
    ⊢ Eq (i.castPred h).rev (i.rev.pred h')
  -/
  rw [← succ_inj, succ_pred, ← rev_castSucc, castSucc_castPred]
  /-
    🎉 no goals
  -/


theorem succ_castPred_eq_castPred_succ {a : Fin (n + 1)} (ha : a ≠ last n)
    (ha' := a.succ_ne_last_iff.mpr ha) :
    (a.castPred ha).succ = (succ a).castPred ha' := rfl


theorem succ_castPred_eq_add_one {a : Fin (n + 1)} (ha : a ≠ last n) :
    (a.castPred ha).succ = a + 1 := by
  /-
    n : Nat
    a : Fin (HAdd.hAdd n 1)
    ha : Ne a (Fin.last n)
    ⊢ Eq (a.castPred ha).succ (HAdd.hAdd a 1)
  -/
  cases' a using lastCases with a
    /-
      case last
      n : Nat
      ha : Ne (Fin.last n) (Fin.last n)
      ⊢ Eq ((Fin.last n).castPred ha).succ (HAdd.hAdd (Fin.last n) 1)
    -/
  · exact (ha rfl).elim
    /-
      🎉 no goals
    -/
    /-
      case cast
      n : Nat
      a : Fin n
      ha : Ne a.castSucc (Fin.last n)
      ⊢ Eq (a.castSucc.castPred ha).succ (HAdd.hAdd a.castSucc 1)
    -/
  · rw [castPred_castSucc, coeSucc_eq_succ]
    /-
      🎉 no goals
    -/


theorem castpred_succ_le_iff {a b : Fin (n + 1)} (ha : succ a ≠ last (n + 1)) :
    (succ a).castPred ha ≤ b ↔ a < b := by
  /-
    n : Nat
    a b : Fin (HAdd.hAdd n 1)
    ha : Ne a.succ (Fin.last (HAdd.hAdd n 1))
    ⊢ Iff (LE.le (a.succ.castPred ha) b) (LT.lt a b)
  -/
  rw [castPred_le_iff, succ_le_castSucc_iff]
  /-
    🎉 no goals
  -/


theorem lt_castPred_succ_iff {a b : Fin (n + 1)} (ha : succ a ≠ last (n + 1)) :
    b < (succ a).castPred ha ↔ b ≤ a := by
  /-
    n : Nat
    a b : Fin (HAdd.hAdd n 1)
    ha : Ne a.succ (Fin.last (HAdd.hAdd n 1))
    ⊢ Iff (LT.lt b (a.succ.castPred ha)) (LE.le b a)
  -/
  rw [lt_castPred_iff, castSucc_lt_succ_iff]
  /-
    🎉 no goals
  -/


theorem lt_castPred_succ {a : Fin (n + 1)} (ha : succ a ≠ last (n + 1)) :
                                   /-
                                     n : Nat
                                     a : Fin (HAdd.hAdd n 1)
                                     ha : Ne a.succ (Fin.last (HAdd.hAdd n 1))
                                     ⊢ LT.lt a (a.succ.castPred ha)
                                   -/
    a < (succ a).castPred ha := by rw [lt_castPred_succ_iff, le_def]
                                   /-
                                     🎉 no goals
                                   -/


theorem succ_castPred_le_iff {a b : Fin (n + 1)} (ha : a ≠ last n) :
    succ (a.castPred ha) ≤ b ↔ a < b := by
  /-
    n : Nat
    a b : Fin (HAdd.hAdd n 1)
    ha : Ne a (Fin.last n)
    ⊢ Iff (LE.le (a.castPred ha).succ b) (LT.lt a b)
  -/
  rw [succ_castPred_eq_castPred_succ ha, castpred_succ_le_iff]
  /-
    🎉 no goals
  -/


theorem lt_succ_castPred_iff {a b : Fin (n + 1)} (ha : a ≠ last n) :
    b < succ (a.castPred ha) ↔ b ≤ a := by
  /-
    n : Nat
    a b : Fin (HAdd.hAdd n 1)
    ha : Ne a (Fin.last n)
    ⊢ Iff (LT.lt b (a.castPred ha).succ) (LE.le b a)
  -/
  rw [succ_castPred_eq_castPred_succ ha, lt_castPred_succ_iff]
  /-
    🎉 no goals
  -/


theorem lt_succ_castPred {a : Fin (n + 1)} (ha : a ≠ last n) :
                                   /-
                                     n : Nat
                                     a : Fin (HAdd.hAdd n 1)
                                     ha : Ne a (Fin.last n)
                                     ⊢ LT.lt a (a.castPred ha).succ
                                   -/
    a < succ (a.castPred ha) := by rw [lt_succ_castPred_iff, le_def]
                                   /-
                                     🎉 no goals
                                   -/


theorem castPred_le_pred_iff {a b : Fin (n + 1)} (ha : a ≠ last n) (hb : b ≠ 0) :
    castPred a ha ≤ pred b hb ↔ a < b := by
  /-
    n : Nat
    a b : Fin (HAdd.hAdd n 1)
    ha : Ne a (Fin.last n)
    hb : Ne b 0
    ⊢ Iff (LE.le (a.castPred ha) (b.pred hb)) (LT.lt a b)
  -/
  rw [le_pred_iff, succ_castPred_le_iff]
  /-
    🎉 no goals
  -/


theorem pred_lt_castPred_iff {a b : Fin (n + 1)} (ha : a ≠ 0) (hb : b ≠ last n) :
    pred a ha < castPred b hb ↔ a ≤ b := by
  /-
    n : Nat
    a b : Fin (HAdd.hAdd n 1)
    ha : Ne a 0
    hb : Ne b (Fin.last n)
    ⊢ Iff (LT.lt (a.pred ha) (b.castPred hb)) (LE.le a b)
  -/
  rw [lt_castPred_iff, castSucc_pred_lt_iff ha]
  /-
    🎉 no goals
  -/


theorem pred_lt_castPred {a : Fin (n + 1)} (h₁ : a ≠ 0) (h₂ : a ≠ last n) :
    pred a h₁ < castPred a h₂ := by
  /-
    n : Nat
    a : Fin (HAdd.hAdd n 1)
    h₁ : Ne a 0
    h₂ : Ne a (Fin.last n)
    ⊢ LT.lt (a.pred h₁) (a.castPred h₂)
  -/
  rw [pred_lt_castPred_iff, le_def]
  /-
    🎉 no goals
  -/


/-- `succAbove p i` embeds `Fin n` into `Fin (n + 1)` with a hole around `p`. -/
def succAbove (p : Fin (n + 1)) (i : Fin n) : Fin (n + 1) :=
  if castSucc i < p then i.castSucc else i.succ


/-- Embedding `i : Fin n` into `Fin (n + 1)` with a hole around `p : Fin (n + 1)`
embeds `i` by `castSucc` when the resulting `i.castSucc < p`. -/
lemma succAbove_of_castSucc_lt (p : Fin (n + 1)) (i : Fin n) (h : castSucc i < p) :
    p.succAbove i = castSucc i := if_pos h


lemma succAbove_of_succ_le (p : Fin (n + 1)) (i : Fin n) (h : succ i ≤ p) :
    p.succAbove i = castSucc i :=
  succAbove_of_castSucc_lt _ _ (castSucc_lt_iff_succ_le.mpr h)


/-- Embedding `i : Fin n` into `Fin (n + 1)` with a hole around `p : Fin (n + 1)`
embeds `i` by `succ` when the resulting `p < i.succ`. -/
lemma succAbove_of_le_castSucc (p : Fin (n + 1)) (i : Fin n) (h : p ≤ castSucc i) :
    p.succAbove i = i.succ := if_neg (Fin.not_lt.2 h)


lemma succAbove_of_lt_succ (p : Fin (n + 1)) (i : Fin n) (h : p < succ i) :
    p.succAbove i = succ i := succAbove_of_le_castSucc _ _ (le_castSucc_iff.mpr h)


lemma succAbove_succ_of_lt (p i : Fin n) (h : p < i) : succAbove p.succ i = i.succ :=
  succAbove_of_lt_succ _ _ (succ_lt_succ_iff.mpr h)


lemma succAbove_succ_of_le (p i : Fin n) (h : i ≤ p) : succAbove p.succ i = i.castSucc :=
  succAbove_of_succ_le _ _ (succ_le_succ_iff.mpr h)


@[simp] lemma succAbove_succ_self (j : Fin n) : j.succ.succAbove j = j.castSucc :=
  succAbove_succ_of_le _ _ Fin.le_rfl


lemma succAbove_castSucc_of_lt (p i : Fin n) (h : i < p) : succAbove p.castSucc i = i.castSucc :=
  succAbove_of_castSucc_lt _ _ (castSucc_lt_castSucc_iff.2 h)


lemma succAbove_castSucc_of_le (p i : Fin n) (h : p ≤ i) : succAbove p.castSucc i = i.succ :=
  succAbove_of_le_castSucc _ _ (castSucc_le_castSucc_iff.2 h)


@[simp] lemma succAbove_castSucc_self (j : Fin n) : succAbove j.castSucc j = j.succ :=
  succAbove_castSucc_of_le _ _ Fin.le_rfl


lemma succAbove_pred_of_lt (p i : Fin (n + 1)) (h : p < i)
    (hi := Fin.ne_of_gt <| Fin.lt_of_le_of_lt p.zero_le h) : succAbove p (i.pred hi) = i := by
  /-
    n : Nat
    p i : Fin (HAdd.hAdd n 1)
    h : LT.lt p i
    hi : optParam (Ne i 0) ⋯
    ⊢ Eq (p.succAbove (i.pred hi)) i
  -/
  rw [succAbove_of_lt_succ _ _ (succ_pred _ _ ▸ h), succ_pred]
  /-
    🎉 no goals
  -/


lemma succAbove_pred_of_le (p i : Fin (n + 1)) (h : i ≤ p) (hi : i ≠ 0) :
    succAbove p (i.pred hi) = (i.pred hi).castSucc := succAbove_of_succ_le _ _ (succ_pred _ _ ▸ h)


@[simp] lemma succAbove_pred_self (p : Fin (n + 1)) (h : p ≠ 0) :
    succAbove p (p.pred h) = (p.pred h).castSucc := succAbove_pred_of_le _ _ Fin.le_rfl h


lemma succAbove_castPred_of_lt (p i : Fin (n + 1)) (h : i < p)
    (hi := Fin.ne_of_lt <| Nat.lt_of_lt_of_le h p.le_last) : succAbove p (i.castPred hi) = i := by
  /-
    n : Nat
    p i : Fin (HAdd.hAdd n 1)
    h : LT.lt i p
    hi : optParam (Ne i (Fin.last n)) ⋯
    ⊢ Eq (p.succAbove (i.castPred hi)) i
  -/
  rw [succAbove_of_castSucc_lt _ _ (castSucc_castPred _ _ ▸ h), castSucc_castPred]
  /-
    🎉 no goals
  -/


lemma succAbove_castPred_of_le (p i : Fin (n + 1)) (h : p ≤ i) (hi : i ≠ last n) :
    succAbove p (i.castPred hi) = (i.castPred hi).succ :=
  succAbove_of_le_castSucc _ _ (castSucc_castPred _ _ ▸ h)


lemma succAbove_castPred_self (p : Fin (n + 1)) (h : p ≠ last n) :
    succAbove p (p.castPred h) = (p.castPred h).succ := succAbove_castPred_of_le _ _ Fin.le_rfl h


lemma succAbove_rev_left (p : Fin (n + 1)) (i : Fin n) :
    p.rev.succAbove i = (p.succAbove i.rev).rev := by
  /-
    n : Nat
    p : Fin (HAdd.hAdd n 1)
    i : Fin n
    ⊢ Eq (p.rev.succAbove i) (p.succAbove i.rev).rev
  -/
  obtain h | h := (rev p).succ_le_or_le_castSucc i
  · rw [succAbove_of_succ_le _ _ h,
      succAbove_of_le_castSucc _ _ (rev_succ _ ▸ (le_rev_iff.mpr h)), rev_succ, rev_rev]
  · rw [succAbove_of_le_castSucc _ _ h,
      succAbove_of_succ_le _ _ (rev_castSucc _ ▸ (rev_le_iff.mpr h)), rev_castSucc, rev_rev]


lemma succAbove_rev_right (p : Fin (n + 1)) (i : Fin n) :
                                                      /-
                                                        n : Nat
                                                        p : Fin (HAdd.hAdd n 1)
                                                        i : Fin n
                                                        ⊢ Eq (p.succAbove i.rev) (p.rev.succAbove i).rev
                                                      -/
    p.succAbove i.rev = (p.rev.succAbove i).rev := by rw [succAbove_rev_left, rev_rev]
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- Embedding `i : Fin n` into `Fin (n + 1)` with a hole around `p : Fin (n + 1)`
never results in `p` itself -/
lemma succAbove_ne (p : Fin (n + 1)) (i : Fin n) : p.succAbove i ≠ p := by
  /-
    n : Nat
    p : Fin (HAdd.hAdd n 1)
    i : Fin n
    ⊢ Ne (p.succAbove i) p
  -/
  rcases p.castSucc_lt_or_lt_succ i with (h | h)
    /-
      case inl
      n : Nat
      p : Fin (HAdd.hAdd n 1)
      i : Fin n
      h : LT.lt i.castSucc p
      ⊢ Ne (p.succAbove i) p
    -/
  · rw [succAbove_of_castSucc_lt _ _ h]
    /-
      case inl
      n : Nat
      p : Fin (HAdd.hAdd n 1)
      i : Fin n
      h : LT.lt i.castSucc p
      ⊢ Ne i.castSucc p
    -/
    exact Fin.ne_of_lt h
    /-
      🎉 no goals
    -/
    /-
      case inr
      n : Nat
      p : Fin (HAdd.hAdd n 1)
      i : Fin n
      h : LT.lt p i.succ
      ⊢ Ne (p.succAbove i) p
    -/
  · rw [succAbove_of_lt_succ _ _ h]
    /-
      case inr
      n : Nat
      p : Fin (HAdd.hAdd n 1)
      i : Fin n
      h : LT.lt p i.succ
      ⊢ Ne i.succ p
    -/
    exact Fin.ne_of_gt h
    /-
      🎉 no goals
    -/


lemma ne_succAbove (p : Fin (n + 1)) (i : Fin n) : p ≠ p.succAbove i := (succAbove_ne _ _).symm


/-- Given a fixed pivot `p : Fin (n + 1)`, `p.succAbove` is injective. -/
lemma succAbove_right_injective : Injective p.succAbove := by
  /-
    n : Nat
    p : Fin (HAdd.hAdd n 1)
    ⊢ Function.Injective p.succAbove
  -/
  rintro i j hij
  /-
    n : Nat
    p : Fin (HAdd.hAdd n 1)
    i j : Fin n
    hij : Eq (p.succAbove i) (p.succAbove j)
    ⊢ Eq i j
  -/
  unfold succAbove at hij
  /-
    n : Nat
    p : Fin (HAdd.hAdd n 1)
    i j : Fin n
    hij : Eq (ite (LT.lt i.castSucc p) i.castSucc i.succ) (ite (LT.lt j.castSucc p …
    ⊢ Eq i j
  -/
  split_ifs at hij with hi hj hj
    /-
      case pos
      n : Nat
      p : Fin (HAdd.hAdd n 1)
      i j : Fin n
      hi : LT.lt i.castSucc p
      hj : LT.lt j.castSucc p
      hij : Eq i.castSucc j.castSucc
      ⊢ Eq i j
    -/
  · exact castSucc_injective _ hij
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      p : Fin (HAdd.hAdd n 1)
      i j : Fin n
      hi : LT.lt i.castSucc p
      hj : Not (LT.lt j.castSucc p)
      hij : Eq i.castSucc j.succ
      ⊢ Eq i j
    -/
  · rw [hij] at hi
    /-
      case neg
      n : Nat
      p : Fin (HAdd.hAdd n 1)
      i j : Fin n
      hi : LT.lt j.succ p
      hj : Not (LT.lt j.castSucc p)
      hij : Eq i.castSucc j.succ
      ⊢ Eq i j
    -/
    cases hj <| Nat.lt_trans j.castSucc_lt_succ hi
    /-
      🎉 no goals
    -/
    /-
      case pos
      n : Nat
      p : Fin (HAdd.hAdd n 1)
      i j : Fin n
      hi : Not (LT.lt i.castSucc p)
      hj : LT.lt j.castSucc p
      hij : Eq i.succ j.castSucc
      ⊢ Eq i j
    -/
  · rw [← hij] at hj
    /-
      case pos
      n : Nat
      p : Fin (HAdd.hAdd n 1)
      i j : Fin n
      hi : Not (LT.lt i.castSucc p)
      hj : LT.lt i.succ p
      hij : Eq i.succ j.castSucc
      ⊢ Eq i j
    -/
    cases hi <| Nat.lt_trans i.castSucc_lt_succ hj
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      p : Fin (HAdd.hAdd n 1)
      i j : Fin n
      hi : Not (LT.lt i.castSucc p)
      hj : Not (LT.lt j.castSucc p)
      hij : Eq i.succ j.succ
      ⊢ Eq i j
    -/
  · exact succ_injective _ hij
    /-
      🎉 no goals
    -/


/-- Given a fixed pivot `p : Fin (n + 1)`, `p.succAbove` is injective. -/
lemma succAbove_right_inj : p.succAbove i = p.succAbove j ↔ i = j :=
  succAbove_right_injective.eq_iff


/--  `Fin.succAbove p` as an `Embedding`. -/
@[simps!]
def succAboveEmb (p : Fin (n + 1)) : Fin n ↪ Fin (n + 1) := ⟨p.succAbove, succAbove_right_injective⟩


@[simp, norm_cast] lemma coe_succAboveEmb (p : Fin (n + 1)) : p.succAboveEmb = p.succAbove := rfl


@[simp]
lemma succAbove_ne_zero_zero [NeZero n] {a : Fin (n + 1)} (ha : a ≠ 0) : a.succAbove 0 = 0 := by
  /-
    n : Nat
    inst✝ : NeZero n
    a : Fin (HAdd.hAdd n 1)
    ha : Ne a 0
    ⊢ Eq (a.succAbove 0) 0
  -/
  rw [Fin.succAbove_of_castSucc_lt]
    /-
      n : Nat
      inst✝ : NeZero n
      a : Fin (HAdd.hAdd n 1)
      ha : Ne a 0
      ⊢ Eq (Fin.castSucc 0) 0
    -/
  · exact castSucc_zero'
    /-
      🎉 no goals
    -/
    /-
      case h
      n : Nat
      inst✝ : NeZero n
      a : Fin (HAdd.hAdd n 1)
      ha : Ne a 0
      ⊢ LT.lt (Fin.castSucc 0) a
    -/
  · exact Fin.pos_iff_ne_zero.2 ha
    /-
      🎉 no goals
    -/


lemma succAbove_eq_zero_iff [NeZero n] {a : Fin (n + 1)} {b : Fin n} (ha : a ≠ 0) :
    a.succAbove b = 0 ↔ b = 0 := by
  /-
    n : Nat
    inst✝ : NeZero n
    a : Fin (HAdd.hAdd n 1)
    b : Fin n
    ha : Ne a 0
    ⊢ Iff (Eq (a.succAbove b) 0) (Eq b 0)
  -/
  rw [← succAbove_ne_zero_zero ha, succAbove_right_inj]
  /-
    🎉 no goals
  -/


lemma succAbove_ne_zero [NeZero n] {a : Fin (n + 1)} {b : Fin n} (ha : a ≠ 0) (hb : b ≠ 0) :
    a.succAbove b ≠ 0 := mt (succAbove_eq_zero_iff ha).mp hb


/-- Embedding `Fin n` into `Fin (n + 1)` with a hole around zero embeds by `succ`. -/
@[simp] lemma succAbove_zero : succAbove (0 : Fin (n + 1)) = Fin.succ := rfl


                                                                      /-
                                                                        n : Nat
                                                                        i : Fin n
                                                                        ⊢ Eq (Fin.succAbove 0 i) i.succ
                                                                      -/
lemma succAbove_zero_apply (i : Fin n) : succAbove 0 i = succ i := by rw [succAbove_zero]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp] lemma succAbove_ne_last_last {a : Fin (n + 2)} (h : a ≠ last (n + 1)) :
    a.succAbove (last n) = last (n + 1) := by
  /-
    n : Nat
    a : Fin (HAdd.hAdd n 2)
    h : Ne a (Fin.last (HAdd.hAdd n 1))
    ⊢ Eq (a.succAbove (Fin.last n)) (Fin.last (HAdd.hAdd n 1))
  -/
  rw [succAbove_of_lt_succ _ _ (succ_last _ ▸ lt_last_iff_ne_last.2 h), succ_last]
  /-
    🎉 no goals
  -/


lemma succAbove_eq_last_iff {a : Fin (n + 2)} {b : Fin (n + 1)} (ha : a ≠ last _) :
    a.succAbove b = last _ ↔ b = last _ := by
  /-
    n : Nat
    a : Fin (HAdd.hAdd n 2)
    b : Fin (HAdd.hAdd n 1)
    ha : Ne a (Fin.last (HAdd.hAdd n 1))
    ⊢ Iff (Eq (a.succAbove b) (Fin.last (HAdd.hAdd n 1))) (Eq b (Fin.last n))
  -/
  simp [← succAbove_ne_last_last ha, succAbove_right_inj]
  /-
    🎉 no goals
  -/


lemma succAbove_ne_last {a : Fin (n + 2)} {b : Fin (n + 1)} (ha : a ≠ last _) (hb : b ≠ last _) :
    a.succAbove b ≠ last _ := mt (succAbove_eq_last_iff ha).mp hb


/-- Embedding `Fin n` into `Fin (n + 1)` with a hole around `last n` embeds by `castSucc`. -/
@[simp] lemma succAbove_last : succAbove (last n) = castSucc := by
  /-
    n : Nat
    ⊢ Eq (Fin.last n).succAbove Fin.castSucc
  -/
  ext; simp only [succAbove_of_castSucc_lt, castSucc_lt_last]
       /-
         🎉 no goals
       -/


                                                                                 /-
                                                                                   n : Nat
                                                                                   i : Fin n
                                                                                   ⊢ Eq ((Fin.last n).succAbove i) i.castSucc
                                                                                 -/
lemma succAbove_last_apply (i : Fin n) : succAbove (last n) i = castSucc i := by rw [succAbove_last]
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


@[deprecated "No deprecation message was provided." (since := "2024-05-30")]
lemma succAbove_lt_ge (p : Fin (n + 1)) (i : Fin n) :
    castSucc i < p ∨ p ≤ castSucc i := Nat.lt_or_ge (castSucc i) p


/-- Embedding `i : Fin n` into `Fin (n + 1)` using a pivot `p` that is greater
results in a value that is less than `p`. -/
lemma succAbove_lt_iff_castSucc_lt (p : Fin (n + 1)) (i : Fin n) :
    p.succAbove i < p ↔ castSucc i < p := by
  /-
    n : Nat
    p : Fin (HAdd.hAdd n 1)
    i : Fin n
    ⊢ Iff (LT.lt (p.succAbove i) p) (LT.lt i.castSucc p)
  -/
  cases' castSucc_lt_or_lt_succ p i with H H
    /-
      case inl
      n : Nat
      p : Fin (HAdd.hAdd n 1)
      i : Fin n
      H : LT.lt i.castSucc p
      ⊢ Iff (LT.lt (p.succAbove i) p) (LT.lt i.castSucc p)
    -/
  · rwa [iff_true_right H, succAbove_of_castSucc_lt _ _ H]
    /-
      🎉 no goals
    -/
    /-
      case inr
      n : Nat
      p : Fin (HAdd.hAdd n 1)
      i : Fin n
      H : LT.lt p i.succ
      ⊢ Iff (LT.lt (p.succAbove i) p) (LT.lt i.castSucc p)
    -/
  · rw [castSucc_lt_iff_succ_le, iff_false_right (Fin.not_le.2 H), succAbove_of_lt_succ _ _ H]
    /-
      case inr
      n : Nat
      p : Fin (HAdd.hAdd n 1)
      i : Fin n
      H : LT.lt p i.succ
      ⊢ Not (LT.lt i.succ p)
    -/
    exact Fin.not_lt.2 <| Fin.le_of_lt H
    /-
      🎉 no goals
    -/


lemma succAbove_lt_iff_succ_le (p : Fin (n + 1)) (i : Fin n) :
    p.succAbove i < p ↔ succ i ≤ p := by
  /-
    n : Nat
    p : Fin (HAdd.hAdd n 1)
    i : Fin n
    ⊢ Iff (LT.lt (p.succAbove i) p) (LE.le i.succ p)
  -/
  rw [succAbove_lt_iff_castSucc_lt, castSucc_lt_iff_succ_le]
  /-
    🎉 no goals
  -/


/-- Embedding `i : Fin n` into `Fin (n + 1)` using a pivot `p` that is lesser
results in a value that is greater than `p`. -/
lemma lt_succAbove_iff_le_castSucc (p : Fin (n + 1)) (i : Fin n) :
    p < p.succAbove i ↔ p ≤ castSucc i := by
  /-
    n : Nat
    p : Fin (HAdd.hAdd n 1)
    i : Fin n
    ⊢ Iff (LT.lt p (p.succAbove i)) (LE.le p i.castSucc)
  -/
  cases' castSucc_lt_or_lt_succ p i with H H
    /-
      case inl
      n : Nat
      p : Fin (HAdd.hAdd n 1)
      i : Fin n
      H : LT.lt i.castSucc p
      ⊢ Iff (LT.lt p (p.succAbove i)) (LE.le p i.castSucc)
    -/
  · rw [iff_false_right (Fin.not_le.2 H), succAbove_of_castSucc_lt _ _ H]
    /-
      case inl
      n : Nat
      p : Fin (HAdd.hAdd n 1)
      i : Fin n
      H : LT.lt i.castSucc p
      ⊢ Not (LT.lt p i.castSucc)
    -/
    exact Fin.not_lt.2 <| Fin.le_of_lt H
    /-
      🎉 no goals
    -/
    /-
      case inr
      n : Nat
      p : Fin (HAdd.hAdd n 1)
      i : Fin n
      H : LT.lt p i.succ
      ⊢ Iff (LT.lt p (p.succAbove i)) (LE.le p i.castSucc)
    -/
  · rwa [succAbove_of_lt_succ _ _ H, iff_true_left H, le_castSucc_iff]
    /-
      🎉 no goals
    -/


lemma lt_succAbove_iff_lt_castSucc (p : Fin (n + 1)) (i : Fin n) :
                                         /-
                                           n : Nat
                                           p : Fin (HAdd.hAdd n 1)
                                           i : Fin n
                                           ⊢ Iff (LT.lt p (p.succAbove i)) (LT.lt p i.succ)
                                         -/
    p < p.succAbove i ↔ p < succ i := by rw [lt_succAbove_iff_le_castSucc, le_castSucc_iff]
                                         /-
                                           🎉 no goals
                                         -/


/-- Embedding a positive `Fin n` results in a positive `Fin (n + 1)` -/
lemma succAbove_pos [NeZero n] (p : Fin (n + 1)) (i : Fin n) (h : 0 < i) : 0 < p.succAbove i := by
  /-
    n : Nat
    inst✝ : NeZero n
    p : Fin (HAdd.hAdd n 1)
    i : Fin n
    h : LT.lt 0 i
    ⊢ LT.lt 0 (p.succAbove i)
  -/
  by_cases H : castSucc i < p
    /-
      case pos
      n : Nat
      inst✝ : NeZero n
      p : Fin (HAdd.hAdd n 1)
      i : Fin n
      h : LT.lt 0 i
      H : LT.lt i.castSucc p
      ⊢ LT.lt 0 (p.succAbove i)
    -/
  · simpa [succAbove_of_castSucc_lt _ _ H] using castSucc_pos' h
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      inst✝ : NeZero n
      p : Fin (HAdd.hAdd n 1)
      i : Fin n
      h : LT.lt 0 i
      H : Not (LT.lt i.castSucc p)
      ⊢ LT.lt 0 (p.succAbove i)
    -/
  · simp [succAbove_of_le_castSucc _ _ (Fin.not_lt.1 H)]
    /-
      🎉 no goals
    -/


lemma castPred_succAbove (x : Fin n) (y : Fin (n + 1)) (h : castSucc x < y)
    (h' := Fin.ne_last_of_lt <| (succAbove_lt_iff_castSucc_lt ..).2 h) :
    (y.succAbove x).castPred h' = x := by
  /-
    n : Nat
    x : Fin n
    y : Fin (HAdd.hAdd n 1)
    h : LT.lt x.castSucc y
    h' : optParam (Ne (y.succAbove x) (Fin.last n)) ⋯
    ⊢ Eq ((y.succAbove x).castPred h') x
  -/
  rw [castPred_eq_iff_eq_castSucc, succAbove_of_castSucc_lt _ _ h]
  /-
    🎉 no goals
  -/


lemma pred_succAbove (x : Fin n) (y : Fin (n + 1)) (h : y ≤ castSucc x)
    (h' := Fin.ne_zero_of_lt <| (lt_succAbove_iff_le_castSucc ..).2 h) :
                                      /-
                                        n : Nat
                                        x : Fin n
                                        y : Fin (HAdd.hAdd n 1)
                                        h : LE.le y x.castSucc
                                        h' : optParam (Ne (y.succAbove x) 0) ⋯
                                        ⊢ Eq ((y.succAbove x).pred h') x
                                      -/
    (y.succAbove x).pred h' = x := by simp only [succAbove_of_le_castSucc _ _ h, pred_succ]
                                      /-
                                        🎉 no goals
                                      -/


lemma exists_succAbove_eq {x y : Fin (n + 1)} (h : x ≠ y) : ∃ z, y.succAbove z = x := by
  /-
    n : Nat
    x y : Fin (HAdd.hAdd n 1)
    h : Ne x y
    ⊢ Exists fun z => Eq (y.succAbove z) x
  -/
  obtain hxy | hyx := Fin.lt_or_lt_of_ne h
  /-
    case inl
    n : Nat
    x y : Fin (HAdd.hAdd n 1)
    h : Ne x y
    hxy : LT.lt x y
    ⊢ Exists fun z => Eq (y.succAbove z) x
  -/
  exacts [⟨_, succAbove_castPred_of_lt _ _ hxy⟩, ⟨_, succAbove_pred_of_lt _ _ hyx⟩]
  /-
    🎉 no goals
  -/


@[simp] lemma exists_succAbove_eq_iff {x y : Fin (n + 1)} : (∃ z, x.succAbove z = y) ↔ y ≠ x :=
      /-
        n : Nat
        x y : Fin (HAdd.hAdd n 1)
        ⊢ (Exists fun z => Eq (x.succAbove z) y) → Ne y x
      -/
  ⟨by rintro ⟨y, rfl⟩; exact succAbove_ne _ _, exists_succAbove_eq⟩
                       /-
                         🎉 no goals
                       -/


/-- The range of `p.succAbove` is everything except `p`. -/
@[simp] lemma range_succAbove (p : Fin (n + 1)) : Set.range p.succAbove = {p}ᶜ :=
  Set.ext fun _ => exists_succAbove_eq_iff


@[simp] lemma range_succ (n : ℕ) : Set.range (Fin.succ : Fin n → Fin (n + 1)) = {0}ᶜ := by
  /-
    n : Nat
    ⊢ Eq (Set.range Fin.succ) (HasCompl.compl (Singleton.singleton 0))
  -/
  rw [← succAbove_zero]; exact range_succAbove (0 : Fin (n + 1))
                         /-
                           🎉 no goals
                         -/


/-- `succAbove` is injective at the pivot -/
lemma succAbove_left_injective : Injective (@succAbove n) := fun _ _ h => by
  /-
    n : Nat
    x✝¹ x✝ : Fin (HAdd.hAdd n 1)
    h : Eq x✝¹.succAbove x✝.succAbove
    ⊢ Eq x✝¹ x✝
  -/
  simpa [range_succAbove] using congr_arg (fun f : Fin n → Fin (n + 1) => (Set.range f)ᶜ) h
  /-
    🎉 no goals
  -/


/-- `succAbove` is injective at the pivot -/
@[simp] lemma succAbove_left_inj {x y : Fin (n + 1)} : x.succAbove = y.succAbove ↔ x = y :=
  succAbove_left_injective.eq_iff


@[simp] lemma zero_succAbove {n : ℕ} (i : Fin n) : (0 : Fin (n + 1)).succAbove i = i.succ := rfl


@[simp] lemma succ_succAbove_zero {n : ℕ} [NeZero n] (i : Fin n) : succAbove i.succ 0 = 0 :=
                                        /-
                                          n : Nat
                                          inst✝ : NeZero n
                                          i : Fin n
                                          ⊢ LT.lt (Fin.castSucc 0) i.succ
                                        -/
  succAbove_of_castSucc_lt i.succ 0 (by simp only [castSucc_zero', succ_pos])
                                        /-
                                          🎉 no goals
                                        -/


/-- `succ` commutes with `succAbove`. -/
@[simp] lemma succ_succAbove_succ {n : ℕ} (i : Fin (n + 1)) (j : Fin n) :
    i.succ.succAbove j.succ = (i.succAbove j).succ := by
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    j : Fin n
    ⊢ Eq (i.succ.succAbove j.succ) (i.succAbove j).succ
  -/
  obtain h | h := i.lt_or_le (succ j)
    /-
      case inl
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      j : Fin n
      h : LT.lt i j.succ
      ⊢ Eq (i.succ.succAbove j.succ) (i.succAbove j).succ
    -/
  · rw [succAbove_of_lt_succ _ _ h, succAbove_succ_of_lt _ _ h]
    /-
      🎉 no goals
    -/
    /-
      case inr
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      j : Fin n
      h : LE.le j.succ i
      ⊢ Eq (i.succ.succAbove j.succ) (i.succAbove j).succ
    -/
  · rwa [succAbove_of_castSucc_lt _ _ h, succAbove_succ_of_le, succ_castSucc]
    /-
      🎉 no goals
    -/


/-- `castSucc` commutes with `succAbove`. -/
@[simp]
lemma castSucc_succAbove_castSucc {n : ℕ} {i : Fin (n + 1)} {j : Fin n} :
    i.castSucc.succAbove j.castSucc = (i.succAbove j).castSucc := by
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    j : Fin n
    ⊢ Eq (i.castSucc.succAbove j.castSucc) (i.succAbove j).castSucc
  -/
  rcases i.le_or_lt (castSucc j) with (h | h)
    /-
      case inl
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      j : Fin n
      h : LE.le i j.castSucc
      ⊢ Eq (i.castSucc.succAbove j.castSucc) (i.succAbove j).castSucc
    -/
  · rw [succAbove_of_le_castSucc _ _ h, succAbove_castSucc_of_le _ _ h, succ_castSucc]
    /-
      🎉 no goals
    -/
    /-
      case inr
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      j : Fin n
      h : LT.lt j.castSucc i
      ⊢ Eq (i.castSucc.succAbove j.castSucc) (i.succAbove j).castSucc
    -/
  · rw [succAbove_of_castSucc_lt _ _ h, succAbove_castSucc_of_lt _ _ h]
    /-
      🎉 no goals
    -/


/-- `pred` commutes with `succAbove`. -/
lemma pred_succAbove_pred {a : Fin (n + 2)} {b : Fin (n + 1)} (ha : a ≠ 0) (hb : b ≠ 0)
    (hk := succAbove_ne_zero ha hb) :
    (a.pred ha).succAbove (b.pred hb) = (a.succAbove b).pred hk := by
  /-
    n : Nat
    a : Fin (HAdd.hAdd n 2)
    b : Fin (HAdd.hAdd n 1)
    ha : Ne a 0
    hb : Ne b 0
    hk : optParam (Ne (a.succAbove b) 0) ⋯
    ⊢ Eq ((a.pred ha).succAbove (b.pred hb)) ((a.succAbove b).pred hk)
  -/
  simp_rw [← succ_inj (b := pred (succAbove a b) hk), ← succ_succAbove_succ, succ_pred]
  /-
    🎉 no goals
  -/


/-- `castPred` commutes with `succAbove`. -/
lemma castPred_succAbove_castPred {a : Fin (n + 2)} {b : Fin (n + 1)} (ha : a ≠ last (n + 1))
    (hb : b ≠ last n) (hk := succAbove_ne_last ha hb) :
    (a.castPred ha).succAbove (b.castPred hb) = (a.succAbove b).castPred hk := by
  simp_rw [← castSucc_inj (b := (a.succAbove b).castPred hk), ← castSucc_succAbove_castSucc,
    castSucc_castPred]


/-- `rev` commutes with `succAbove`. -/
lemma rev_succAbove (p : Fin (n + 1)) (i : Fin n) :
    rev (succAbove p i) = succAbove (rev p) (rev i) := by
  /-
    n : Nat
    p : Fin (HAdd.hAdd n 1)
    i : Fin n
    ⊢ Eq (p.succAbove i).rev (p.rev.succAbove i.rev)
  -/
  rw [succAbove_rev_left, rev_rev]
  /-
    🎉 no goals
  -/

--@[simp] -- Porting note: can be proved by `simp`

lemma one_succAbove_zero {n : ℕ} : (1 : Fin (n + 2)).succAbove 0 = 0 := by
  /-
    n : Nat
    ⊢ Eq (Fin.succAbove 1 0) 0
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- By moving `succ` to the outside of this expression, we create opportunities for further
simplification using `succAbove_zero` or `succ_succAbove_zero`. -/
@[simp] lemma succ_succAbove_one {n : ℕ} [NeZero n] (i : Fin (n + 1)) :
    i.succ.succAbove 1 = (i.succAbove 0).succ := by
  /-
    n : Nat
    inst✝ : NeZero n
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (i.succ.succAbove 1) (i.succAbove 0).succ
  -/
  rw [← succ_zero_eq_one']; convert succ_succAbove_succ i 0
                            /-
                              🎉 no goals
                            -/


@[simp] lemma one_succAbove_succ {n : ℕ} (j : Fin n) :
    (1 : Fin (n + 2)).succAbove j.succ = j.succ.succ := by
  /-
    n : Nat
    j : Fin n
    ⊢ Eq (Fin.succAbove 1 j.succ) j.succ.succ
  -/
  have := succ_succAbove_succ 0 j; rwa [succ_zero_eq_one, zero_succAbove] at this
                                   /-
                                     🎉 no goals
                                   -/


@[simp] lemma one_succAbove_one {n : ℕ} : (1 : Fin (n + 3)).succAbove 1 = 2 := by
  simpa only [succ_zero_eq_one, val_zero, zero_succAbove, succ_one_eq_two]
    using succ_succAbove_succ (0 : Fin (n + 2)) (0 : Fin (n + 2))


/-- `predAbove p i` surjects `i : Fin (n+1)` into `Fin n` by subtracting one if `p < i`. -/
def predAbove (p : Fin n) (i : Fin (n + 1)) : Fin n :=
  if h : castSucc p < i
  then pred i (Fin.ne_zero_of_lt h)
  else castPred i (Fin.ne_of_lt <| Fin.lt_of_le_of_lt (Fin.not_lt.1 h) (castSucc_lt_last _))


lemma predAbove_of_le_castSucc (p : Fin n) (i : Fin (n + 1)) (h : i ≤ castSucc p)
    (hi := Fin.ne_of_lt <| Fin.lt_of_le_of_lt h <| castSucc_lt_last _) :
    p.predAbove i = i.castPred hi := dif_neg <| Fin.not_lt.2 h


lemma predAbove_of_lt_succ (p : Fin n) (i : Fin (n + 1)) (h : i < succ p)
    (hi := Fin.ne_last_of_lt h) : p.predAbove i = i.castPred hi :=
  predAbove_of_le_castSucc _ _ (le_castSucc_iff.mpr h)


lemma predAbove_of_castSucc_lt (p : Fin n) (i : Fin (n + 1)) (h : castSucc p < i)
    (hi := Fin.ne_zero_of_lt h) : p.predAbove i = i.pred hi := dif_pos h


lemma predAbove_of_succ_le (p : Fin n) (i : Fin (n + 1)) (h : succ p ≤ i)
    (hi := Fin.ne_of_gt <| Fin.lt_of_lt_of_le (succ_pos _) h) :
    p.predAbove i = i.pred hi := predAbove_of_castSucc_lt _ _ (castSucc_lt_iff_succ_le.mpr h)


lemma predAbove_succ_of_lt (p i : Fin n) (h : i < p) (hi := succ_ne_last_of_lt h) :
    p.predAbove (succ i) = (i.succ).castPred hi := by
  /-
    n : Nat
    p i : Fin n
    h : LT.lt i p
    hi : optParam (Ne i.succ (Fin.last n)) ⋯
    ⊢ Eq (p.predAbove i.succ) (i.succ.castPred hi)
  -/
  rw [predAbove_of_lt_succ _ _ (succ_lt_succ_iff.mpr h)]
  /-
    🎉 no goals
  -/


lemma predAbove_succ_of_le (p i : Fin n) (h : p ≤ i) : p.predAbove (succ i) = i := by
  /-
    n : Nat
    p i : Fin n
    h : LE.le p i
    ⊢ Eq (p.predAbove i.succ) i
  -/
  rw [predAbove_of_succ_le _ _ (succ_le_succ_iff.mpr h), pred_succ]
  /-
    🎉 no goals
  -/


@[simp] lemma predAbove_succ_self (p : Fin n) : p.predAbove (succ p) = p :=
  predAbove_succ_of_le _ _ Fin.le_rfl


lemma predAbove_castSucc_of_lt (p i : Fin n) (h : p < i) (hi := castSucc_ne_zero_of_lt h) :
    p.predAbove (castSucc i) = i.castSucc.pred hi := by
  /-
    n : Nat
    p i : Fin n
    h : LT.lt p i
    hi : optParam (Ne i.castSucc 0) ⋯
    ⊢ Eq (p.predAbove i.castSucc) (i.castSucc.pred hi)
  -/
  rw [predAbove_of_castSucc_lt _ _ (castSucc_lt_castSucc_iff.2 h)]
  /-
    🎉 no goals
  -/


lemma predAbove_castSucc_of_le (p i : Fin n) (h : i ≤ p) : p.predAbove (castSucc i) = i := by
  /-
    n : Nat
    p i : Fin n
    h : LE.le i p
    ⊢ Eq (p.predAbove i.castSucc) i
  -/
  rw [predAbove_of_le_castSucc _ _ (castSucc_le_castSucc_iff.mpr h), castPred_castSucc]
  /-
    🎉 no goals
  -/


@[simp] lemma predAbove_castSucc_self (p : Fin n) : p.predAbove (castSucc p) = p :=
  predAbove_castSucc_of_le _ _ Fin.le_rfl


lemma predAbove_pred_of_lt (p i : Fin (n + 1)) (h : i < p) (hp := Fin.ne_zero_of_lt h)
    (hi := Fin.ne_last_of_lt h) : (pred p hp).predAbove i = castPred i hi := by
  /-
    n : Nat
    p i : Fin (HAdd.hAdd n 1)
    h : LT.lt i p
    hp : optParam (Ne p 0) ⋯
    hi : optParam (Ne i (Fin.last n)) ⋯
    ⊢ Eq ((p.pred hp).predAbove i) (i.castPred hi)
  -/
  rw [predAbove_of_lt_succ _ _ (succ_pred _ _ ▸ h)]
  /-
    🎉 no goals
  -/


lemma predAbove_pred_of_le (p i : Fin (n + 1)) (h : p ≤ i) (hp : p ≠ 0)
    (hi := Fin.ne_of_gt <| Fin.lt_of_lt_of_le (Fin.pos_iff_ne_zero.2 hp) h) :
                                            /-
                                              n : Nat
                                              p i : Fin (HAdd.hAdd n 1)
                                              h : LE.le p i
                                              hp : Ne p 0
                                              hi : optParam (Ne i 0) ⋯
                                              ⊢ Eq ((p.pred hp).predAbove i) (i.pred hi)
                                            -/
  (pred p hp).predAbove i = pred i hi := by rw [predAbove_of_succ_le _ _ (succ_pred _ _ ▸ h)]
                                            /-
                                              🎉 no goals
                                            -/


lemma predAbove_pred_self (p : Fin (n + 1)) (hp : p ≠ 0) : (pred p hp).predAbove p = pred p hp :=
  predAbove_pred_of_le _ _ Fin.le_rfl hp


lemma predAbove_castPred_of_lt (p i : Fin (n + 1)) (h : p < i) (hp := Fin.ne_last_of_lt h)
  (hi := Fin.ne_zero_of_lt h) : (castPred p hp).predAbove i = pred i hi := by
  /-
    n : Nat
    p i : Fin (HAdd.hAdd n 1)
    h : LT.lt p i
    hp : optParam (Ne p (Fin.last n)) ⋯
    hi : optParam (Ne i 0) ⋯
    ⊢ Eq ((p.castPred hp).predAbove i) (i.pred hi)
  -/
  rw [predAbove_of_castSucc_lt _ _ (castSucc_castPred _ _ ▸ h)]
  /-
    🎉 no goals
  -/


lemma predAbove_castPred_of_le (p i : Fin (n + 1)) (h : i ≤ p) (hp : p ≠ last n)
    (hi := Fin.ne_of_lt <| Fin.lt_of_le_of_lt h <| Fin.lt_last_iff_ne_last.2 hp) :
    (castPred p hp).predAbove i = castPred i hi := by
  /-
    n : Nat
    p i : Fin (HAdd.hAdd n 1)
    h : LE.le i p
    hp : Ne p (Fin.last n)
    hi : optParam (Ne i (Fin.last n)) ⋯
    ⊢ Eq ((p.castPred hp).predAbove i) (i.castPred hi)
  -/
  rw [predAbove_of_le_castSucc _ _ (castSucc_castPred _ _ ▸ h)]
  /-
    🎉 no goals
  -/


lemma predAbove_castPred_self (p : Fin (n + 1)) (hp : p ≠ last n) :
    (castPred p hp).predAbove p = castPred p hp := predAbove_castPred_of_le _ _ Fin.le_rfl hp


lemma predAbove_rev_left (p : Fin n) (i : Fin (n + 1)) :
    p.rev.predAbove i = (p.predAbove i.rev).rev := by
  /-
    n : Nat
    p : Fin n
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (p.rev.predAbove i) (p.predAbove i.rev).rev
  -/
  obtain h | h := (rev i).succ_le_or_le_castSucc p
  · rw [predAbove_of_succ_le _ _ h, rev_pred,
      predAbove_of_le_castSucc _ _ (rev_succ _ ▸ (le_rev_iff.mpr h)), castPred_inj, rev_rev]
  · rw [predAbove_of_le_castSucc _ _ h, rev_castPred,
      predAbove_of_succ_le _ _ (rev_castSucc _ ▸ (rev_le_iff.mpr h)), pred_inj, rev_rev]


lemma predAbove_rev_right (p : Fin n) (i : Fin (n + 1)) :
                                                      /-
                                                        n : Nat
                                                        p : Fin n
                                                        i : Fin (HAdd.hAdd n 1)
                                                        ⊢ Eq (p.predAbove i.rev) (p.rev.predAbove i).rev
                                                      -/
    p.predAbove i.rev = (p.rev.predAbove i).rev := by rw [predAbove_rev_left, rev_rev]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp] lemma predAbove_right_zero [NeZero n] {i : Fin n} : predAbove (i : Fin n) 0 = 0 := by
  /-
    n : Nat
    inst✝ : NeZero n
    i : Fin n
    ⊢ Eq (i.predAbove 0) 0
  -/
  cases n
    /-
      case zero
      inst✝ : NeZero 0
      i : Fin 0
      ⊢ Eq (i.predAbove 0) 0
    -/
  · exact i.elim0
    /-
      🎉 no goals
    -/
    /-
      case succ
      n✝ : Nat
      inst✝ : NeZero (HAdd.hAdd n✝ 1)
      i : Fin (HAdd.hAdd n✝ 1)
      ⊢ Eq (i.predAbove 0) 0
    -/
  · rw [predAbove_of_le_castSucc _ _ (zero_le _), castPred_zero]
    /-
      🎉 no goals
    -/


@[simp] lemma predAbove_zero_succ [NeZero n] {i : Fin n} : predAbove 0 i.succ = i := by
  /-
    n : Nat
    inst✝ : NeZero n
    i : Fin n
    ⊢ Eq (Fin.predAbove 0 i.succ) i
  -/
  rw [predAbove_succ_of_le _ _ (Fin.zero_le' _)]
  /-
    🎉 no goals
  -/


@[simp]
lemma succ_predAbove_zero [NeZero n] {j : Fin (n + 1)} (h : j ≠ 0) : succ (predAbove 0 j) = j := by
  /-
    n : Nat
    inst✝ : NeZero n
    j : Fin (HAdd.hAdd n 1)
    h : Ne j 0
    ⊢ Eq (Fin.predAbove 0 j).succ j
  -/
  rcases exists_succ_eq_of_ne_zero h with ⟨k, rfl⟩
  /-
    case intro
    n : Nat
    inst✝ : NeZero n
    k : Fin n
    h : Ne k.succ 0
    ⊢ Eq (Fin.predAbove 0 k.succ).succ k.succ
  -/
  rw [predAbove_zero_succ]
  /-
    🎉 no goals
  -/


@[simp] lemma predAbove_zero_of_ne_zero [NeZero n] {i : Fin (n + 1)} (hi : i ≠ 0) :
    predAbove 0 i = i.pred hi := by
  /-
    n : Nat
    inst✝ : NeZero n
    i : Fin (HAdd.hAdd n 1)
    hi : Ne i 0
    ⊢ Eq (Fin.predAbove 0 i) (i.pred hi)
  -/
  obtain ⟨y, rfl⟩ := exists_succ_eq.2 hi; exact predAbove_zero_succ
                                          /-
                                            🎉 no goals
                                          -/


lemma predAbove_zero [NeZero n] {i : Fin (n + 1)} :
    predAbove (0 : Fin n) i = if hi : i = 0 then 0 else i.pred hi := by
  /-
    n : Nat
    inst✝ : NeZero n
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (Fin.predAbove 0 i) (dite (Eq i 0) (fun hi => 0) fun hi => i.pred hi)
  -/
  split_ifs with hi
    /-
      case pos
      n : Nat
      inst✝ : NeZero n
      i : Fin (HAdd.hAdd n 1)
      hi : Eq i 0
      ⊢ Eq (Fin.predAbove 0 i) 0
    -/
  · rw [hi, predAbove_right_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      inst✝ : NeZero n
      i : Fin (HAdd.hAdd n 1)
      hi : Not (Eq i 0)
      ⊢ Eq (Fin.predAbove 0 i) (i.pred hi)
    -/
  · rw [predAbove_zero_of_ne_zero hi]
    /-
      🎉 no goals
    -/


@[simp] lemma predAbove_right_last {i : Fin (n + 1)} : predAbove i (last (n + 1)) = last n := by
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (i.predAbove (Fin.last (HAdd.hAdd n 1))) (Fin.last n)
  -/
  rw [predAbove_of_castSucc_lt _ _ (castSucc_lt_last _), pred_last]
  /-
    🎉 no goals
  -/


@[simp] lemma predAbove_last_castSucc {i : Fin (n + 1)} : predAbove (last n) (i.castSucc) = i := by
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq ((Fin.last n).predAbove i.castSucc) i
  -/
  rw [predAbove_of_le_castSucc _ _ (castSucc_le_castSucc_iff.mpr (le_last _)), castPred_castSucc]
  /-
    🎉 no goals
  -/


@[simp] lemma predAbove_last_of_ne_last {i : Fin (n + 2)} (hi : i ≠ last (n + 1)) :
    predAbove (last n) i = castPred i hi := by
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    hi : Ne i (Fin.last (HAdd.hAdd n 1))
    ⊢ Eq ((Fin.last n).predAbove i) (i.castPred hi)
  -/
  rw [← exists_castSucc_eq] at hi
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    hi✝ : Ne i (Fin.last (HAdd.hAdd n 1))
    hi : Exists fun j => Eq j.castSucc i
    ⊢ Eq ((Fin.last n).predAbove i) (i.castPred hi✝)
  -/
  rcases hi with ⟨y, rfl⟩
  /-
    case intro
    n : Nat
    y : Fin (HAdd.hAdd n 1)
    hi : Ne y.castSucc (Fin.last (HAdd.hAdd n 1))
    ⊢ Eq ((Fin.last n).predAbove y.castSucc) (y.castSucc.castPred hi)
  -/
  exact predAbove_last_castSucc
  /-
    🎉 no goals
  -/


lemma predAbove_last_apply {i : Fin (n + 2)} :
    predAbove (last n) i = if hi : i = last _ then last _ else i.castPred hi := by
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    ⊢ Eq ((Fin.last n).predAbove i) (dite (Eq i (Fin.last (HAdd.hAdd n 1))) (fun h …
  -/
  split_ifs with hi
    /-
      case pos
      n : Nat
      i : Fin (HAdd.hAdd n 2)
      hi : Eq i (Fin.last (HAdd.hAdd n 1))
      ⊢ Eq ((Fin.last n).predAbove i) (Fin.last n)
    -/
  · rw [hi, predAbove_right_last]
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      i : Fin (HAdd.hAdd n 2)
      hi : Not (Eq i (Fin.last (HAdd.hAdd n 1)))
      ⊢ Eq ((Fin.last n).predAbove i) (i.castPred hi)
    -/
  · rw [predAbove_last_of_ne_last hi]
    /-
      🎉 no goals
    -/


/-- Sending `Fin (n+1)` to `Fin n` by subtracting one from anything above `p`
then back to `Fin (n+1)` with a gap around `p` is the identity away from `p`. -/
@[simp]
lemma succAbove_predAbove {p : Fin n} {i : Fin (n + 1)} (h : i ≠ castSucc p) :
    p.castSucc.succAbove (p.predAbove i) = i := by
  /-
    n : Nat
    p : Fin n
    i : Fin (HAdd.hAdd n 1)
    h : Ne i p.castSucc
    ⊢ Eq (p.castSucc.succAbove (p.predAbove i)) i
  -/
  obtain h | h := Fin.lt_or_lt_of_ne h
    /-
      case inl
      n : Nat
      p : Fin n
      i : Fin (HAdd.hAdd n 1)
      h✝ : Ne i p.castSucc
      h : LT.lt i p.castSucc
      ⊢ Eq (p.castSucc.succAbove (p.predAbove i)) i
    -/
  · rw [predAbove_of_le_castSucc _ _ (Fin.le_of_lt h), succAbove_castPred_of_lt _ _ h]
    /-
      🎉 no goals
    -/
    /-
      case inr
      n : Nat
      p : Fin n
      i : Fin (HAdd.hAdd n 1)
      h✝ : Ne i p.castSucc
      h : LT.lt p.castSucc i
      ⊢ Eq (p.castSucc.succAbove (p.predAbove i)) i
    -/
  · rw [predAbove_of_castSucc_lt _ _ h, succAbove_pred_of_lt _ _ h]
    /-
      🎉 no goals
    -/


/-- Sending `Fin n` into `Fin (n + 1)` with a gap at `p`
then back to `Fin n` by subtracting one from anything above `p` is the identity. -/
@[simp]
lemma predAbove_succAbove (p : Fin n) (i : Fin n) : p.predAbove ((castSucc p).succAbove i) = i := by
  /-
    n : Nat
    p i : Fin n
    ⊢ Eq (p.predAbove (p.castSucc.succAbove i)) i
  -/
  obtain h | h := p.le_or_lt i
    /-
      case inl
      n : Nat
      p i : Fin n
      h : LE.le p i
      ⊢ Eq (p.predAbove (p.castSucc.succAbove i)) i
    -/
  · rw [succAbove_castSucc_of_le _ _ h, predAbove_succ_of_le _ _ h]
    /-
      🎉 no goals
    -/
    /-
      case inr
      n : Nat
      p i : Fin n
      h : LT.lt i p
      ⊢ Eq (p.predAbove (p.castSucc.succAbove i)) i
    -/
  · rw [succAbove_castSucc_of_lt _ _ h, predAbove_castSucc_of_le _ _ <| Fin.le_of_lt h]
    /-
      🎉 no goals
    -/


/-- `succ` commutes with `predAbove`. -/
@[simp] lemma succ_predAbove_succ (a : Fin n) (b : Fin (n + 1)) :
    a.succ.predAbove b.succ = (a.predAbove b).succ := by
  /-
    n : Nat
    a : Fin n
    b : Fin (HAdd.hAdd n 1)
    ⊢ Eq (a.succ.predAbove b.succ) (a.predAbove b).succ
  -/
  obtain h | h := Fin.le_or_lt (succ a) b
    /-
      case inl
      n : Nat
      a : Fin n
      b : Fin (HAdd.hAdd n 1)
      h : LE.le a.succ b
      ⊢ Eq (a.succ.predAbove b.succ) (a.predAbove b).succ
    -/
  · rw [predAbove_of_castSucc_lt _ _ h, predAbove_succ_of_le _ _ h, succ_pred]
    /-
      🎉 no goals
    -/
    /-
      case inr
      n : Nat
      a : Fin n
      b : Fin (HAdd.hAdd n 1)
      h : LT.lt b a.succ
      ⊢ Eq (a.succ.predAbove b.succ) (a.predAbove b).succ
    -/
  · rw [predAbove_of_lt_succ _ _ h, predAbove_succ_of_lt _ _ h, succ_castPred_eq_castPred_succ]
    /-
      🎉 no goals
    -/


/-- `castSucc` commutes with `predAbove`. -/
@[simp] lemma castSucc_predAbove_castSucc {n : ℕ} (a : Fin n) (b : Fin (n + 1)) :
    a.castSucc.predAbove b.castSucc = (a.predAbove b).castSucc := by
  /-
    n : Nat
    a : Fin n
    b : Fin (HAdd.hAdd n 1)
    ⊢ Eq (a.castSucc.predAbove b.castSucc) (a.predAbove b).castSucc
  -/
  obtain h | h := a.castSucc.lt_or_le b
  · rw [predAbove_of_castSucc_lt _ _ h, predAbove_castSucc_of_lt _ _ h,
      castSucc_pred_eq_pred_castSucc]
    /-
      case inr
      n : Nat
      a : Fin n
      b : Fin (HAdd.hAdd n 1)
      h : LE.le b a.castSucc
      ⊢ Eq (a.castSucc.predAbove b.castSucc) (a.predAbove b).castSucc
    -/
  · rw [predAbove_of_le_castSucc _ _ h, predAbove_castSucc_of_le _ _ h, castSucc_castPred]
    /-
      🎉 no goals
    -/


/-- `rev` commutes with `predAbove`. -/
lemma rev_predAbove {n : ℕ} (p : Fin n) (i : Fin (n + 1)) :
                                                      /-
                                                        n : Nat
                                                        p : Fin n
                                                        i : Fin (HAdd.hAdd n 1)
                                                        ⊢ Eq (p.predAbove i).rev (p.rev.predAbove i.rev)
                                                      -/
    (predAbove p i).rev = predAbove p.rev i.rev := by rw [predAbove_rev_left, rev_rev]
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- Compute `i / n`, where `n` is a `Nat` and inferred the type of `i`. -/
def divNat (i : Fin (m * n)) : Fin m :=
  ⟨i / n, Nat.div_lt_of_lt_mul <| Nat.mul_comm m n ▸ i.prop⟩


@[simp]
theorem coe_divNat (i : Fin (m * n)) : (i.divNat : ℕ) = i / n :=
  rfl


/-- Compute `i % n`, where `n` is a `Nat` and inferred the type of `i`. -/
def modNat (i : Fin (m * n)) : Fin n := ⟨i % n, Nat.mod_lt _ <| Nat.pos_of_mul_pos_left i.pos⟩


@[simp]
theorem coe_modNat (i : Fin (m * n)) : (i.modNat : ℕ) = i % n :=
  rfl


theorem modNat_rev (i : Fin (m * n)) : i.rev.modNat = i.modNat.rev := by
  /-
    n m : Nat
    i : Fin (HMul.hMul m n)
    ⊢ Eq i.rev.modNat i.modNat.rev
  -/
  ext
  /-
    case h
    n m : Nat
    i : Fin (HMul.hMul m n)
    ⊢ Eq ↑i.rev.modNat ↑i.modNat.rev
  -/
  have H₁ : i % n + 1 ≤ n := i.modNat.is_lt
  /-
    case h
    n m : Nat
    i : Fin (HMul.hMul m n)
    H₁ : LE.le (HAdd.hAdd (HMod.hMod (↑i) n) 1) n
    ⊢ Eq ↑i.rev.modNat ↑i.modNat.rev
  -/
  have H₂ : i / n < m := i.divNat.is_lt
  /-
    case h
    n m : Nat
    i : Fin (HMul.hMul m n)
    H₁ : LE.le (HAdd.hAdd (HMod.hMod (↑i) n) 1) n
    H₂ : LT.lt (HDiv.hDiv (↑i) n) m
    ⊢ Eq ↑i.rev.modNat ↑i.modNat.rev
  -/
  simp only [coe_modNat, val_rev]
  calc
    (m * n - (i + 1)) % n = (m * n - ((i / n) * n + i % n + 1)) % n := by rw [Nat.div_add_mod']
    _ = ((m - i / n - 1) * n + (n - (i % n + 1))) % n := by
      rw [Nat.mul_sub_right_distrib, Nat.one_mul, Nat.sub_add_sub_cancel _ H₁,
        Nat.mul_sub_right_distrib, Nat.sub_sub, Nat.add_assoc]
      exact Nat.le_mul_of_pos_left _ <| Nat.le_sub_of_add_le' H₂
    _ = n - (i % n + 1) := by
      rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt]; exact i.modNat.rev.is_lt


theorem liftFun_iff_succ {α : Type*} (r : α → α → Prop) [IsTrans α r] {f : Fin (n + 1) → α} :
    ((· < ·) ⇒ r) f f ↔ ∀ i : Fin n, r (f (castSucc i)) (f i.succ) := by
  /-
    n : Nat
    α : Type u_1
    r : α → α → Prop
    inst✝ : IsTrans α r
    f : Fin (HAdd.hAdd n 1) → α
    ⊢ Iff (Relator.LiftFun (fun x1 x2 => LT.lt x1 x2) r f f) (∀ (i : Fin n), r (f  …
  -/
  constructor
    /-
      case mp
      n : Nat
      α : Type u_1
      r : α → α → Prop
      inst✝ : IsTrans α r
      f : Fin (HAdd.hAdd n 1) → α
      ⊢ Relator.LiftFun (fun x1 x2 => LT.lt x1 x2) r f f → ∀ (i : Fin n), r (f i.cas …
    -/
  · intro H i
    /-
      case mp
      n : Nat
      α : Type u_1
      r : α → α → Prop
      inst✝ : IsTrans α r
      f : Fin (HAdd.hAdd n 1) → α
      H : Relator.LiftFun (fun x1 x2 => LT.lt x1 x2) r f f
      i : Fin n
      ⊢ r (f i.castSucc) (f i.succ)
    -/
    exact H i.castSucc_lt_succ
    /-
      🎉 no goals
    -/
    /-
      case mpr
      n : Nat
      α : Type u_1
      r : α → α → Prop
      inst✝ : IsTrans α r
      f : Fin (HAdd.hAdd n 1) → α
      ⊢ (∀ (i : Fin n), r (f i.castSucc) (f i.succ)) → Relator.LiftFun (fun x1 x2 => …
    -/
  · refine fun H i => Fin.induction (fun h ↦ ?_) ?_
      /-
        case mpr.refine_1
        n : Nat
        α : Type u_1
        r : α → α → Prop
        inst✝ : IsTrans α r
        f : Fin (HAdd.hAdd n 1) → α
        H : ∀ (i : Fin n), r (f i.castSucc) (f i.succ)
        i : Fin (HAdd.hAdd n 1)
        h : (fun x1 x2 => LT.lt x1 x2) i 0
        ⊢ r (f i) (f 0)
      -/
    · simp [le_def] at h
      /-
        🎉 no goals
      -/
      /-
        case mpr.refine_2
        n : Nat
        α : Type u_1
        r : α → α → Prop
        inst✝ : IsTrans α r
        f : Fin (HAdd.hAdd n 1) → α
        H : ∀ (i : Fin n), r (f i.castSucc) (f i.succ)
        i : Fin (HAdd.hAdd n 1)
        ⊢ ∀ (i_1 : Fin n), ((fun x1 x2 => LT.lt x1 x2) i i_1.castSucc → r (f i) (f i_1 …
      -/
    · intro j ihj hij
      /-
        case mpr.refine_2
        n : Nat
        α : Type u_1
        r : α → α → Prop
        inst✝ : IsTrans α r
        f : Fin (HAdd.hAdd n 1) → α
        H : ∀ (i : Fin n), r (f i.castSucc) (f i.succ)
        i : Fin (HAdd.hAdd n 1)
        j : Fin n
        ihj : (fun x1 x2 => LT.lt x1 x2) i j.castSucc → r (f i) (f j.castSucc)
        hij : LT.lt i j.succ
        ⊢ r (f i) (f j.succ)
      -/
      rw [← le_castSucc_iff] at hij
      /-
        case mpr.refine_2
        n : Nat
        α : Type u_1
        r : α → α → Prop
        inst✝ : IsTrans α r
        f : Fin (HAdd.hAdd n 1) → α
        H : ∀ (i : Fin n), r (f i.castSucc) (f i.succ)
        i : Fin (HAdd.hAdd n 1)
        j : Fin n
        ihj : (fun x1 x2 => LT.lt x1 x2) i j.castSucc → r (f i) (f j.castSucc)
        hij : LE.le i j.castSucc
        ⊢ r (f i) (f j.succ)
      -/
      obtain hij | hij := (le_def.1 hij).eq_or_lt
        /-
          case mpr.refine_2.inl
          n : Nat
          α : Type u_1
          r : α → α → Prop
          inst✝ : IsTrans α r
          f : Fin (HAdd.hAdd n 1) → α
          H : ∀ (i : Fin n), r (f i.castSucc) (f i.succ)
          i : Fin (HAdd.hAdd n 1)
          j : Fin n
          ihj : (fun x1 x2 => LT.lt x1 x2) i j.castSucc → r (f i) (f j.castSucc)
          hij✝ : LE.le i j.castSucc
          hij : Eq ↑i ↑j.castSucc
          ⊢ r (f i) (f j.succ)
        -/
      · obtain rfl := Fin.ext hij
        /-
          case mpr.refine_2.inl
          n : Nat
          α : Type u_1
          r : α → α → Prop
          inst✝ : IsTrans α r
          f : Fin (HAdd.hAdd n 1) → α
          H : ∀ (i : Fin n), r (f i.castSucc) (f i.succ)
          j : Fin n
          ihj : (fun x1 x2 => LT.lt x1 x2) j.castSucc j.castSucc → r (f j.castSucc) (f j …
          hij✝ : LE.le j.castSucc j.castSucc
          hij : Eq ↑j.castSucc ↑j.castSucc
          ⊢ r (f j.castSucc) (f j.succ)
        -/
        exact H _
        /-
          🎉 no goals
        -/
        /-
          case mpr.refine_2.inr
          n : Nat
          α : Type u_1
          r : α → α → Prop
          inst✝ : IsTrans α r
          f : Fin (HAdd.hAdd n 1) → α
          H : ∀ (i : Fin n), r (f i.castSucc) (f i.succ)
          i : Fin (HAdd.hAdd n 1)
          j : Fin n
          ihj : (fun x1 x2 => LT.lt x1 x2) i j.castSucc → r (f i) (f j.castSucc)
          hij✝ : LE.le i j.castSucc
          hij : LT.lt ↑i ↑j.castSucc
          ⊢ r (f i) (f j.succ)
        -/
      · exact _root_.trans (ihj hij) (H j)
        /-
          🎉 no goals
        -/


/-- Negation on `Fin n` -/
instance neg (n : ℕ) : Neg (Fin n) :=
  ⟨fun a => ⟨(n - a) % n, Nat.mod_lt _ a.pos⟩⟩


theorem neg_def (a : Fin n) : -a = ⟨(n - a) % n, Nat.mod_lt _ a.pos⟩ := rfl


protected theorem coe_neg (a : Fin n) : ((-a : Fin n) : ℕ) = (n - a) % n :=
  rfl


theorem eq_zero (n : Fin 1) : n = 0 := Subsingleton.elim _ _


@[deprecated val_eq_zero (since := "2024-09-18")]
                                                    /-
                                                      a : Fin 1
                                                      ⊢ Eq (↑a) 0
                                                    -/
theorem coe_fin_one (a : Fin 1) : (a : ℕ) = 0 := by simp [Subsingleton.elim a 0]
                                                    /-
                                                      🎉 no goals
                                                    -/


lemma eq_one_of_neq_zero (i : Fin 2) (hi : i ≠ 0) : i = 1 := by
  /-
    i : Fin 2
    hi : Ne i 0
    ⊢ Eq i 1
  -/
  fin_omega
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_neg_one : ↑(-1 : Fin (n + 1)) = n := by
  /-
    n : Nat
    ⊢ Eq (↑(-1)) n
  -/
  cases n
    /-
      case zero
      ⊢ Eq (↑(-1)) 0
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case succ
    n✝ : Nat
    ⊢ Eq (↑(-1)) (HAdd.hAdd n✝ 1)
  -/
  rw [Fin.coe_neg, Fin.val_one, Nat.add_one_sub_one, Nat.mod_eq_of_lt]
  /-
    case succ
    n✝ : Nat
    ⊢ LT.lt (HAdd.hAdd n✝ 1) (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
  -/
  constructor
  /-
    🎉 no goals
  -/


theorem last_sub (i : Fin (n + 1)) : last n - i = Fin.rev i :=
                /-
                  n : Nat
                  i : Fin (HAdd.hAdd n 1)
                  ⊢ Eq ↑(HSub.hSub (Fin.last n) i) ↑i.rev
                -/
  Fin.ext <| by rw [coe_sub_iff_le.2 i.le_last, val_last, val_rev, Nat.succ_sub_succ_eq_sub]
                /-
                  🎉 no goals
                -/


theorem add_one_le_of_lt {n : ℕ} {a b : Fin (n + 1)} (h : a < b) : a + 1 ≤ b := by
  /-
    n : Nat
    a b : Fin (HAdd.hAdd n 1)
    h : LT.lt a b
    ⊢ LE.le (HAdd.hAdd a 1) b
  -/
              /-
                🎉 no goals
              -/
  cases n <;> fin_omega
              /-
                🎉 no goals
              -/


theorem exists_eq_add_of_le {n : ℕ} {a b : Fin n} (h : a ≤ b) : ∃ k ≤ b, b = a + k := by
  /-
    n : Nat
    a b : Fin n
    h : LE.le a b
    ⊢ Exists fun k => And (LE.le k b) (Eq b (HAdd.hAdd a k))
  -/
  obtain ⟨k, hk⟩ : ∃ k : ℕ, (b : ℕ) = a + k := Nat.exists_eq_add_of_le h
  /-
    case intro
    n : Nat
    a b : Fin n
    h : LE.le a b
    k : Nat
    hk : Eq (↑b) (HAdd.hAdd (↑a) k)
    ⊢ Exists fun k => And (LE.le k b) (Eq b (HAdd.hAdd a k))
  -/
  have hkb : k ≤ b := by omega
  /-
    case intro
    n : Nat
    a b : Fin n
    h : LE.le a b
    k : Nat
    hk : Eq (↑b) (HAdd.hAdd (↑a) k)
    hkb : LE.le k ↑b
    ⊢ Exists fun k => And (LE.le k b) (Eq b (HAdd.hAdd a k))
  -/
  refine ⟨⟨k, hkb.trans_lt b.is_lt⟩, hkb, ?_⟩
  /-
    case intro
    n : Nat
    a b : Fin n
    h : LE.le a b
    k : Nat
    hk : Eq (↑b) (HAdd.hAdd (↑a) k)
    hkb : LE.le k ↑b
    ⊢ Eq b (HAdd.hAdd a ⟨k, ⋯⟩)
  -/
  simp [Fin.ext_iff, Fin.val_add, ← hk, Nat.mod_eq_of_lt b.is_lt]
  /-
    🎉 no goals
  -/


theorem exists_eq_add_of_lt {n : ℕ} {a b : Fin (n + 1)} (h : a < b) :
    ∃ k < b, k + 1 ≤ b ∧ b = a + k + 1 := by
  /-
    n : Nat
    a b : Fin (HAdd.hAdd n 1)
    h : LT.lt a b
    ⊢ Exists fun k => And (LT.lt k b) (And (LE.le (HAdd.hAdd k 1) b) (Eq b (HAdd.h …
  -/
  cases n
    /-
      case zero
      a b : Fin (HAdd.hAdd 0 1)
      h : LT.lt a b
      ⊢ Exists fun k => And (LT.lt k b) (And (LE.le (HAdd.hAdd k 1) b) (Eq b (HAdd.h …
    -/
  · omega
    /-
      🎉 no goals
    -/
  /-
    case succ
    n✝ : Nat
    a b : Fin (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
    h : LT.lt a b
    ⊢ Exists fun k => And (LT.lt k b) (And (LE.le (HAdd.hAdd k 1) b) (Eq b (HAdd.h …
  -/
  obtain ⟨k, hk⟩ : ∃ k : ℕ, (b : ℕ) = a + k + 1 := Nat.exists_eq_add_of_lt h
  /-
    case succ.intro
    n✝ : Nat
    a b : Fin (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
    h : LT.lt a b
    k : Nat
    hk : Eq (↑b) (HAdd.hAdd (HAdd.hAdd (↑a) k) 1)
    ⊢ Exists fun k => And (LT.lt k b) (And (LE.le (HAdd.hAdd k 1) b) (Eq b (HAdd.h …
  -/
  have hkb : k < b := by omega
  /-
    case succ.intro
    n✝ : Nat
    a b : Fin (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
    h : LT.lt a b
    k : Nat
    hk : Eq (↑b) (HAdd.hAdd (HAdd.hAdd (↑a) k) 1)
    hkb : LT.lt k ↑b
    ⊢ Exists fun k => And (LT.lt k b) (And (LE.le (HAdd.hAdd k 1) b) (Eq b (HAdd.h …
  -/
  refine ⟨⟨k, hkb.trans b.is_lt⟩, hkb, by fin_omega, ?_⟩
  /-
    case succ.intro
    n✝ : Nat
    a b : Fin (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
    h : LT.lt a b
    k : Nat
    hk : Eq (↑b) (HAdd.hAdd (HAdd.hAdd (↑a) k) 1)
    hkb : LT.lt k ↑b
    ⊢ Eq b (HAdd.hAdd (HAdd.hAdd a ⟨k, ⋯⟩) 1)
  -/
  simp [Fin.ext_iff, Fin.val_add, ← hk, Nat.mod_eq_of_lt b.is_lt]
  /-
    🎉 no goals
  -/


lemma pos_of_ne_zero {n : ℕ} {a : Fin (n + 1)} (h : a ≠ 0) :
    0 < a :=
  Nat.pos_of_ne_zero (val_ne_of_ne h)


lemma sub_succ_le_sub_of_le {n : ℕ} {u v : Fin (n + 2)} (h : u < v) : v - (u + 1) < v - u := by
  /-
    n : Nat
    u v : Fin (HAdd.hAdd n 2)
    h : LT.lt u v
    ⊢ LT.lt (HSub.hSub v (HAdd.hAdd u 1)) (HSub.hSub v u)
  -/
  fin_omega
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_natCast_eq_mod (m n : ℕ) [NeZero m] :
    ((n : Fin m) : ℕ) = n % m :=
  rfl


theorem coe_ofNat_eq_mod (m n : ℕ) [NeZero m] :
    ((ofNat(n) : Fin m) : ℕ) = ofNat(n) % m :=
  rfl


protected theorem mul_one' [NeZero n] (k : Fin n) : k * 1 = k := by
  /-
    n : Nat
    inst✝ : NeZero n
    k : Fin n
    ⊢ Eq (HMul.hMul k 1) k
  -/
  cases' n with n
    /-
      case zero
      inst✝ : NeZero 0
      k : Fin 0
      ⊢ Eq (HMul.hMul k 1) k
    -/
  · simp [eq_iff_true_of_subsingleton]
    /-
      🎉 no goals
    -/
  /-
    case succ
    n : Nat
    inst✝ : NeZero (HAdd.hAdd n 1)
    k : Fin (HAdd.hAdd n 1)
    ⊢ Eq (HMul.hMul k 1) k
  -/
  cases n
    /-
      case succ.zero
      inst✝ : NeZero (HAdd.hAdd 0 1)
      k : Fin (HAdd.hAdd 0 1)
      ⊢ Eq (HMul.hMul k 1) k
    -/
  · simp [fin_one_eq_zero]
    /-
      🎉 no goals
    -/
  /-
    case succ.succ
    n✝ : Nat
    inst✝ : NeZero (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
    k : Fin (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
    ⊢ Eq (HMul.hMul k 1) k
  -/
  simp [Fin.ext_iff, mul_def, mod_eq_of_lt (is_lt k)]
  /-
    🎉 no goals
  -/


protected theorem one_mul' [NeZero n] (k : Fin n) : (1 : Fin n) * k = k := by
  /-
    n : Nat
    inst✝ : NeZero n
    k : Fin n
    ⊢ Eq (HMul.hMul 1 k) k
  -/
  rw [Fin.mul_comm, Fin.mul_one']
  /-
    🎉 no goals
  -/


                                                                     /-
                                                                       n : Nat
                                                                       inst✝ : NeZero n
                                                                       k : Fin n
                                                                       ⊢ Eq (HMul.hMul k 0) 0
                                                                     -/
protected theorem mul_zero' [NeZero n] (k : Fin n) : k * 0 = 0 := by simp [Fin.ext_iff, mul_def]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


protected theorem zero_mul' [NeZero n] (k : Fin n) : (0 : Fin n) * k = 0 := by
  /-
    n : Nat
    inst✝ : NeZero n
    k : Fin n
    ⊢ Eq (HMul.hMul 0 k) 0
  -/
  simp [Fin.ext_iff, mul_def]
  /-
    🎉 no goals
  -/


