private def lbp (m n : ℕ) : Prop :=
  m = n + 1 ∧ ∀ k ≤ n, ¬p k


private def wf_lbp : WellFounded (@lbp p) :=
  ⟨let ⟨n, pn⟩ := H
    suffices ∀ m k, n ≤ k + m → Acc lbp k from fun _ => this _ _ (Nat.le_add_left _ _)
    fun m =>
    Nat.recOn m
      (fun _ kn =>
        ⟨_, fun y r =>
          match y, r with
          | _, ⟨rfl, a⟩ => absurd pn (a _ kn)⟩)
      fun m IH k kn =>
      ⟨_, fun y r =>
        match y, r with
                                   /-
                                     m✝¹ n✝ k✝ : Nat
                                     p q : Nat → Prop
                                     inst✝ : DecidablePred p
                                     H : Exists fun n => p n
                                     n : Nat
                                     pn : p n
                                     m✝ m : Nat
                                     IH : ∀ (k : Nat), LE.le n (HAdd.hAdd k m) → Acc Nat.lbp k
                                     k : Nat
                                     kn : LE.le n (HAdd.hAdd k m.succ)
                                     y : Nat
                                     r : Nat.lbp y k
                                     _a : ∀ (k_1 : Nat), LE.le k_1 k → Not (p k_1)
                                     ⊢ LE.le n (HAdd.hAdd (HAdd.hAdd k 1) m)
                                   -/
        | _, ⟨rfl, _a⟩ => IH _ (by rw [Nat.add_right_comm]; exact kn)⟩⟩
                                                            /-
                                                              🎉 no goals
                                                            -/


protected def findX : { n // p n ∧ ∀ m < n, ¬p m } :=
  @WellFounded.fix _ (fun k => (∀ n < k, ¬p n) → { n // p n ∧ ∀ m < n, ¬p m }) lbp (wf_lbp H)
    (fun m IH al =>
      if pm : p m then ⟨m, pm, al⟩
      else
        have : ∀ n ≤ m, ¬p n := fun n h =>
                                                            /-
                                                              m✝ n✝ k : Nat
                                                              p q : Nat → Prop
                                                              inst✝ : DecidablePred p
                                                              H : Exists fun n => p n
                                                              m : Nat
                                                              IH : (y : Nat) → Nat.lbp y m → (fun k => (∀ (n : Nat), LT.lt n k → Not (p n))  …
                                                              al : ∀ (n : Nat), LT.lt n m → Not (p n)
                                                              pm : Not (p m)
                                                              n : Nat
                                                              h : LE.le n m
                                                              e : Eq n m
                                                              ⊢ Not (p n)
                                                            -/
          Or.elim (Nat.lt_or_eq_of_le h) (al n) fun e => by rw [e]; exact pm
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
        IH _ ⟨rfl, this⟩ fun n h => this n <| Nat.le_of_succ_le_succ h)
    0 fun _ h => absurd h (Nat.not_lt_zero _)


/-- If `p` is a (decidable) predicate on `ℕ` and `hp : ∃ (n : ℕ), p n` is a proof that
there exists some natural number satisfying `p`, then `Nat.find hp` is the
smallest natural number satisfying `p`. Note that `Nat.find` is protected,
meaning that you can't just write `find`, even if the `Nat` namespace is open.

The API for `Nat.find` is:

* `Nat.find_spec` is the proof that `Nat.find hp` satisfies `p`.
* `Nat.find_min` is the proof that if `m < Nat.find hp` then `m` does not satisfy `p`.
* `Nat.find_min'` is the proof that if `m` does satisfy `p` then `Nat.find hp ≤ m`.
-/
protected def find : ℕ :=
  (Nat.findX H).1


protected theorem find_spec : p (Nat.find H) :=
  (Nat.findX H).2.left


protected theorem find_min : ∀ {m : ℕ}, m < Nat.find H → ¬p m :=
  @(Nat.findX H).2.right


protected theorem find_min' {m : ℕ} (h : p m) : Nat.find H ≤ m :=
  Nat.le_of_not_lt fun l => Nat.find_min H l h


lemma find_eq_iff (h : ∃ n : ℕ, p n) : Nat.find h = m ↔ p m ∧ ∀ n < m, ¬ p n := by
  /-
    m : Nat
    p : Nat → Prop
    inst✝ : DecidablePred p
    h : Exists fun n => p n
    ⊢ Iff (Eq (Nat.find h) m) (And (p m) (∀ (n : Nat), LT.lt n m → Not (p n)))
  -/
  constructor
    /-
      case mp
      m : Nat
      p : Nat → Prop
      inst✝ : DecidablePred p
      h : Exists fun n => p n
      ⊢ Eq (Nat.find h) m → And (p m) (∀ (n : Nat), LT.lt n m → Not (p n))
    -/
  · rintro rfl
    /-
      case mp
      p : Nat → Prop
      inst✝ : DecidablePred p
      h : Exists fun n => p n
      ⊢ And (p (Nat.find h)) (∀ (n : Nat), LT.lt n (Nat.find h) → Not (p n))
    -/
    exact ⟨Nat.find_spec h, fun _ ↦ Nat.find_min h⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      m : Nat
      p : Nat → Prop
      inst✝ : DecidablePred p
      h : Exists fun n => p n
      ⊢ And (p m) (∀ (n : Nat), LT.lt n m → Not (p n)) → Eq (Nat.find h) m
    -/
  · rintro ⟨hm, hlt⟩
    /-
      case mpr.intro
      m : Nat
      p : Nat → Prop
      inst✝ : DecidablePred p
      h : Exists fun n => p n
      hm : p m
      hlt : ∀ (n : Nat), LT.lt n m → Not (p n)
      ⊢ Eq (Nat.find h) m
    -/
    exact le_antisymm (Nat.find_min' h hm) (not_lt.1 <| imp_not_comm.1 (hlt _) <| Nat.find_spec h)
    /-
      🎉 no goals
    -/


@[simp] lemma find_lt_iff (h : ∃ n : ℕ, p n) (n : ℕ) : Nat.find h < n ↔ ∃ m < n, p m :=
  ⟨fun h2 ↦ ⟨Nat.find h, h2, Nat.find_spec h⟩,
    fun ⟨_, hmn, hm⟩ ↦ Nat.lt_of_le_of_lt (Nat.find_min' h hm) hmn⟩


@[simp] lemma find_le_iff (h : ∃ n : ℕ, p n) (n : ℕ) : Nat.find h ≤ n ↔ ∃ m ≤ n, p m := by
  /-
    p : Nat → Prop
    inst✝ : DecidablePred p
    h : Exists fun n => p n
    n : Nat
    ⊢ Iff (LE.le (Nat.find h) n) (Exists fun m => And (LE.le m n) (p m))
  -/
  simp only [exists_prop, ← Nat.lt_succ_iff, find_lt_iff]
  /-
    🎉 no goals
  -/


@[simp] lemma le_find_iff (h : ∃ n : ℕ, p n) (n : ℕ) : n ≤ Nat.find h ↔ ∀ m < n, ¬ p m := by
  /-
    p : Nat → Prop
    inst✝ : DecidablePred p
    h : Exists fun n => p n
    n : Nat
    ⊢ Iff (LE.le n (Nat.find h)) (∀ (m : Nat), LT.lt m n → Not (p m))
  -/
  simp only [← not_lt, find_lt_iff, not_exists, not_and]
  /-
    🎉 no goals
  -/


@[simp] lemma lt_find_iff (h : ∃ n : ℕ, p n) (n : ℕ) : n < Nat.find h ↔ ∀ m ≤ n, ¬ p m := by
  /-
    p : Nat → Prop
    inst✝ : DecidablePred p
    h : Exists fun n => p n
    n : Nat
    ⊢ Iff (LT.lt n (Nat.find h)) (∀ (m : Nat), LE.le m n → Not (p m))
  -/
  simp only [← succ_le_iff, le_find_iff, succ_le_succ_iff]
  /-
    🎉 no goals
  -/


                                                                           /-
                                                                             p : Nat → Prop
                                                                             inst✝ : DecidablePred p
                                                                             h : Exists fun n => p n
                                                                             ⊢ Iff (Eq (Nat.find h) 0) (p 0)
                                                                           -/
@[simp] lemma find_eq_zero (h : ∃ n : ℕ, p n) : Nat.find h = 0 ↔ p 0 := by simp [find_eq_iff]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


variable [DecidablePred q] in
lemma find_mono (h : ∀ n, q n → p n) {hp : ∃ n, p n} {hq : ∃ n, q n} : Nat.find hp ≤ Nat.find hq :=
  Nat.find_min' _ (h _ (Nat.find_spec hq))


lemma find_le {h : ∃ n, p n} (hn : p n) : Nat.find h ≤ n :=
  (Nat.find_le_iff _ _).2 ⟨n, le_refl _, hn⟩


lemma find_comp_succ (h₁ : ∃ n, p n) (h₂ : ∃ n, p (n + 1)) (h0 : ¬ p 0) :
    Nat.find h₁ = Nat.find h₂ + 1 := by
  /-
    p : Nat → Prop
    inst✝ : DecidablePred p
    h₁ : Exists fun n => p n
    h₂ : Exists fun n => p (HAdd.hAdd n 1)
    h0 : Not (p 0)
    ⊢ Eq (Nat.find h₁) (HAdd.hAdd (Nat.find h₂) 1)
  -/
  refine (find_eq_iff _).2 ⟨Nat.find_spec h₂, fun n hn ↦ ?_⟩
  /-
    p : Nat → Prop
    inst✝ : DecidablePred p
    h₁ : Exists fun n => p n
    h₂ : Exists fun n => p (HAdd.hAdd n 1)
    h0 : Not (p 0)
    n : Nat
    hn : LT.lt n (HAdd.hAdd (Nat.find h₂) 1)
    ⊢ Not (p n)
  -/
  cases n
  /-
    case zero
    p : Nat → Prop
    inst✝ : DecidablePred p
    h₁ : Exists fun n => p n
    h₂ : Exists fun n => p (HAdd.hAdd n 1)
    h0 : Not (p 0)
    hn : LT.lt 0 (HAdd.hAdd (Nat.find h₂) 1)
    ⊢ Not (p 0)
  -/
  exacts [h0, @Nat.find_min (fun n ↦ p (n + 1)) _ h₂ _ (succ_lt_succ_iff.1 hn)]
  /-
    🎉 no goals
  -/


lemma find_pos (h : ∃ n : ℕ, p n) : 0 < Nat.find h ↔ ¬p 0 :=
  Nat.pos_iff_ne_zero.trans (Nat.find_eq_zero _).not


lemma find_add {hₘ : ∃ m, p (m + n)} {hₙ : ∃ n, p n} (hn : n ≤ Nat.find hₙ) :
    Nat.find hₘ + n = Nat.find hₙ := by
  /-
    n : Nat
    p : Nat → Prop
    inst✝ : DecidablePred p
    hₘ : Exists fun m => p (HAdd.hAdd m n)
    hₙ : Exists fun n => p n
    hn : LE.le n (Nat.find hₙ)
    ⊢ Eq (HAdd.hAdd (Nat.find hₘ) n) (Nat.find hₙ)
  -/
  refine le_antisymm ((le_find_iff _ _).2 fun m hm hpm => Nat.not_le.2 hm ?_) ?_
    /-
      case refine_1
      n : Nat
      p : Nat → Prop
      inst✝ : DecidablePred p
      hₘ : Exists fun m => p (HAdd.hAdd m n)
      hₙ : Exists fun n => p n
      hn : LE.le n (Nat.find hₙ)
      m : Nat
      hm : LT.lt m (HAdd.hAdd (Nat.find hₘ) n)
      hpm : p m
      ⊢ LE.le (HAdd.hAdd (Nat.find hₘ) n) m
    -/
  · have hnm : n ≤ m := le_trans hn (find_le hpm)
    /-
      case refine_1
      n : Nat
      p : Nat → Prop
      inst✝ : DecidablePred p
      hₘ : Exists fun m => p (HAdd.hAdd m n)
      hₙ : Exists fun n => p n
      hn : LE.le n (Nat.find hₙ)
      m : Nat
      hm : LT.lt m (HAdd.hAdd (Nat.find hₘ) n)
      hpm : p m
      hnm : LE.le n m
      ⊢ LE.le (HAdd.hAdd (Nat.find hₘ) n) m
    -/
    refine Nat.add_le_of_le_sub hnm (find_le ?_)
    /-
      case refine_1
      n : Nat
      p : Nat → Prop
      inst✝ : DecidablePred p
      hₘ : Exists fun m => p (HAdd.hAdd m n)
      hₙ : Exists fun n => p n
      hn : LE.le n (Nat.find hₙ)
      m : Nat
      hm : LT.lt m (HAdd.hAdd (Nat.find hₘ) n)
      hpm : p m
      hnm : LE.le n m
      ⊢ p (HAdd.hAdd (HSub.hSub m n) n)
    -/
    rwa [Nat.sub_add_cancel hnm]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      n : Nat
      p : Nat → Prop
      inst✝ : DecidablePred p
      hₘ : Exists fun m => p (HAdd.hAdd m n)
      hₙ : Exists fun n => p n
      hn : LE.le n (Nat.find hₙ)
      ⊢ LE.le (Nat.find hₙ) (HAdd.hAdd (Nat.find hₘ) n)
    -/
  · rw [← Nat.sub_le_iff_le_add]
    /-
      case refine_2
      n : Nat
      p : Nat → Prop
      inst✝ : DecidablePred p
      hₘ : Exists fun m => p (HAdd.hAdd m n)
      hₙ : Exists fun n => p n
      hn : LE.le n (Nat.find hₙ)
      ⊢ LE.le (HSub.hSub (Nat.find hₙ) n) (Nat.find hₘ)
    -/
    refine (le_find_iff _ _).2 fun m hm hpm => Nat.not_le.2 hm ?_
    /-
      case refine_2
      n : Nat
      p : Nat → Prop
      inst✝ : DecidablePred p
      hₘ : Exists fun m => p (HAdd.hAdd m n)
      hₙ : Exists fun n => p n
      hn : LE.le n (Nat.find hₙ)
      m : Nat
      hm : LT.lt m (HSub.hSub (Nat.find hₙ) n)
      hpm : p (HAdd.hAdd m n)
      ⊢ LE.le (HSub.hSub (Nat.find hₙ) n) m
    -/
    rw [Nat.sub_le_iff_le_add]
    /-
      case refine_2
      n : Nat
      p : Nat → Prop
      inst✝ : DecidablePred p
      hₘ : Exists fun m => p (HAdd.hAdd m n)
      hₙ : Exists fun n => p n
      hn : LE.le n (Nat.find hₙ)
      m : Nat
      hm : LT.lt m (HSub.hSub (Nat.find hₙ) n)
      hpm : p (HAdd.hAdd m n)
      ⊢ LE.le (Nat.find hₙ) (HAdd.hAdd m n)
    -/
    exact find_le hpm
    /-
      🎉 no goals
    -/


/-- `Nat.findGreatest P n` is the largest `i ≤ bound` such that `P i` holds, or `0` if no such `i`
exists -/
def findGreatest (P : ℕ → Prop) [DecidablePred P] : ℕ → ℕ
  | 0 => 0
  | n + 1 => if P (n + 1) then n + 1 else Nat.findGreatest P n


@[simp] lemma findGreatest_zero : Nat.findGreatest P 0 = 0 := rfl


lemma findGreatest_succ (n : ℕ) :
    Nat.findGreatest P (n + 1) = if P (n + 1) then n + 1 else Nat.findGreatest P n := rfl


@[simp] lemma findGreatest_eq : ∀ {n}, P n → Nat.findGreatest P n = n
  | 0, _ => rfl
                   /-
                     P : Nat → Prop
                     inst✝ : DecidablePred P
                     n : Nat
                     h : P (HAdd.hAdd n 1)
                     ⊢ Eq (Nat.findGreatest P (HAdd.hAdd n 1)) (HAdd.hAdd n 1)
                   -/
  | n + 1, h => by simp [Nat.findGreatest, h]
                   /-
                     🎉 no goals
                   -/


@[simp]
lemma findGreatest_of_not (h : ¬ P (n + 1)) : findGreatest P (n + 1) = findGreatest P n := by
  /-
    P : Nat → Prop
    inst✝ : DecidablePred P
    n : Nat
    h : Not (P (HAdd.hAdd n 1))
    ⊢ Eq (Nat.findGreatest P (HAdd.hAdd n 1)) (Nat.findGreatest P n)
  -/
  simp [Nat.findGreatest, h]
  /-
    🎉 no goals
  -/


lemma findGreatest_eq_iff :
    Nat.findGreatest P k = m ↔ m ≤ k ∧ (m ≠ 0 → P m) ∧ ∀ ⦃n⦄, m < n → n ≤ k → ¬P n := by
  induction k generalizing m with
  | zero =>
    rw [eq_comm, Iff.comm]
    simp only [zero_eq, Nat.le_zero, ne_eq, findGreatest_zero, and_iff_left_iff_imp]
    rintro rfl
    exact ⟨fun h ↦ (h rfl).elim, fun n hlt heq ↦ by omega⟩
  | succ k ihk =>
    by_cases hk : P (k + 1)
    · rw [findGreatest_eq hk]
      constructor
      · rintro rfl
        exact ⟨le_refl _, fun _ ↦ hk, fun n hlt hle ↦ by omega⟩
      · rintro ⟨hle, h0, hm⟩
        rcases Decidable.eq_or_lt_of_le hle with (rfl | hlt)
        exacts [rfl, (hm hlt (le_refl _) hk).elim]
    · rw [findGreatest_of_not hk, ihk]
      constructor
      · rintro ⟨hle, hP, hm⟩
        refine ⟨le_trans hle k.le_succ, hP, fun n hlt hle ↦ ?_⟩
        rcases Decidable.eq_or_lt_of_le hle with (rfl | hlt')
        exacts [hk, hm hlt <| Nat.lt_succ_iff.1 hlt']
      · rintro ⟨hle, hP, hm⟩
        refine ⟨Nat.lt_succ_iff.1 (lt_of_le_of_ne hle ?_), hP,
          fun n hlt hle ↦ hm hlt (le_trans hle k.le_succ)⟩
        rintro rfl
        exact hk (hP k.succ_ne_zero)


lemma findGreatest_eq_zero_iff : Nat.findGreatest P k = 0 ↔ ∀ ⦃n⦄, 0 < n → n ≤ k → ¬P n := by
  /-
    k : Nat
    P : Nat → Prop
    inst✝ : DecidablePred P
    ⊢ Iff (Eq (Nat.findGreatest P k) 0) (∀ ⦃n : Nat⦄, LT.lt 0 n → LE.le n k → Not  …
  -/
  simp [findGreatest_eq_iff]
  /-
    🎉 no goals
  -/


@[simp] lemma findGreatest_pos : 0 < Nat.findGreatest P k ↔ ∃ n, 0 < n ∧ n ≤ k ∧ P n := by
  /-
    k : Nat
    P : Nat → Prop
    inst✝ : DecidablePred P
    ⊢ Iff (LT.lt 0 (Nat.findGreatest P k)) (Exists fun n => And (LT.lt 0 n) (And ( …
  -/
  rw [Nat.pos_iff_ne_zero, Ne, findGreatest_eq_zero_iff]; push_neg; rfl
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


lemma findGreatest_spec (hmb : m ≤ n) (hm : P m) : P (Nat.findGreatest P n) := by
  /-
    m : Nat
    P : Nat → Prop
    inst✝ : DecidablePred P
    n : Nat
    hmb : LE.le m n
    hm : P m
    ⊢ P (Nat.findGreatest P n)
  -/
  by_cases h : Nat.findGreatest P n = 0
    /-
      case pos
      m : Nat
      P : Nat → Prop
      inst✝ : DecidablePred P
      n : Nat
      hmb : LE.le m n
      hm : P m
      h : Eq (Nat.findGreatest P n) 0
      ⊢ P (Nat.findGreatest P n)
    -/
  · cases m
      /-
        case pos.zero
        P : Nat → Prop
        inst✝ : DecidablePred P
        n : Nat
        h : Eq (Nat.findGreatest P n) 0
        hmb : LE.le 0 n
        hm : P 0
        ⊢ P (Nat.findGreatest P n)
      -/
    · rwa [h]
      /-
        🎉 no goals
      -/
    /-
      case pos.succ
      P : Nat → Prop
      inst✝ : DecidablePred P
      n : Nat
      h : Eq (Nat.findGreatest P n) 0
      n✝ : Nat
      hmb : LE.le (HAdd.hAdd n✝ 1) n
      hm : P (HAdd.hAdd n✝ 1)
      ⊢ P (Nat.findGreatest P n)
    -/
    exact ((findGreatest_eq_zero_iff.1 h) (zero_lt_succ _) hmb hm).elim
    /-
      🎉 no goals
    -/
    /-
      case neg
      m : Nat
      P : Nat → Prop
      inst✝ : DecidablePred P
      n : Nat
      hmb : LE.le m n
      hm : P m
      h : Not (Eq (Nat.findGreatest P n) 0)
      ⊢ P (Nat.findGreatest P n)
    -/
  · exact (findGreatest_eq_iff.1 rfl).2.1 h
    /-
      🎉 no goals
    -/


lemma findGreatest_le (n : ℕ) : Nat.findGreatest P n ≤ n :=
  (findGreatest_eq_iff.1 rfl).1


lemma le_findGreatest (hmb : m ≤ n) (hm : P m) : m ≤ Nat.findGreatest P n :=
  le_of_not_lt fun hlt => (findGreatest_eq_iff.1 rfl).2.2 hlt hmb hm


lemma findGreatest_mono_right (P : ℕ → Prop) [DecidablePred P] {m n} (hmn : m ≤ n) :
    Nat.findGreatest P m ≤ Nat.findGreatest P n := by
  induction hmn with
  | refl => simp
  | step hmk ih =>
    rw [findGreatest_succ]
    split_ifs
    · exact le_trans ih <| le_trans (findGreatest_le _) (le_succ _)
    · exact ih


lemma findGreatest_mono_left [DecidablePred Q] (hPQ : ∀ n, P n → Q n) (n : ℕ) :
    Nat.findGreatest P n ≤ Nat.findGreatest Q n := by
  induction n with
  | zero => rfl
  | succ n hn =>
    by_cases h : P (n + 1)
    · rw [findGreatest_eq h, findGreatest_eq (hPQ _ h)]
    · rw [findGreatest_of_not h]
      exact le_trans hn (Nat.findGreatest_mono_right _ <| le_succ _)


lemma findGreatest_mono [DecidablePred Q] (hPQ : ∀ n, P n → Q n) (hmn : m ≤ n) :
    Nat.findGreatest P m ≤ Nat.findGreatest Q n :=
  le_trans (Nat.findGreatest_mono_right _ hmn) (findGreatest_mono_left hPQ _)


theorem findGreatest_is_greatest (hk : Nat.findGreatest P n < k) (hkb : k ≤ n) : ¬P k :=
  (findGreatest_eq_iff.1 rfl).2.2 hk hkb


theorem findGreatest_of_ne_zero (h : Nat.findGreatest P n = m) (h0 : m ≠ 0) : P m :=
  (findGreatest_eq_iff.1 h).2.1 h0


