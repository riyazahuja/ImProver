instance decidablePredExistsNat : DecidablePred fun n' : ℕ => ∃ (n : ℕ+) (_ : n' = n), p n :=
  fun n' =>
  decidable_of_iff' (∃ h : 0 < n', p ⟨n', h⟩) <|
    Subtype.exists.trans <| by
      /-
        p q : PNat → Prop
        inst✝¹ : DecidablePred p
        inst✝ : DecidablePred q
        h : Exists fun n => p n
        n' : Nat
        ⊢ Iff (Exists fun a => Exists fun b => Exists fun x => p ⟨a, b⟩) (Exists fun h …
      -/
      simp_rw [mk_coe, @exists_comm (_ < _) (_ = _), exists_prop, exists_eq_left']
      /-
        🎉 no goals
      -/


/-- The `PNat` version of `Nat.findX` -/
protected def findX : { n // p n ∧ ∀ m : ℕ+, m < n → ¬p m } := by
  /-
    p q : PNat → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidablePred q
    h : Exists fun n => p n
    ⊢ Subtype fun n => And (p n) (∀ (m : PNat), LT.lt m n → Not (p m))
  -/
  have : ∃ (n' : ℕ) (n : ℕ+) (_ : n' = n), p n := Exists.elim h fun n hn => ⟨n, n, rfl, hn⟩
  /-
    p q : PNat → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidablePred q
    h : Exists fun n => p n
    this : Exists fun n' => Exists fun n => Exists fun x => p n
    ⊢ Subtype fun n => And (p n) (∀ (m : PNat), LT.lt m n → Not (p m))
  -/
  have n := Nat.findX this
  /-
    p q : PNat → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidablePred q
    h : Exists fun n => p n
    this : Exists fun n' => Exists fun n => Exists fun x => p n
    n : Subtype fun n => And (Exists fun n_1 => Exists fun x => p n_1) (∀ (m : Nat …
    ⊢ Subtype fun n => And (p n) (∀ (m : PNat), LT.lt m n → Not (p m))
  -/
  refine ⟨⟨n, ?_⟩, ?_, fun m hm pm => ?_⟩
    /-
      case refine_1
      p q : PNat → Prop
      inst✝¹ : DecidablePred p
      inst✝ : DecidablePred q
      h : Exists fun n => p n
      this : Exists fun n' => Exists fun n => Exists fun x => p n
      n : Subtype fun n => And (Exists fun n_1 => Exists fun x => p n_1) (∀ (m : Nat …
      ⊢ LT.lt 0 ↑n
    -/
  · obtain ⟨n', hn', -⟩ := n.prop.1
    /-
      case refine_1.intro.intro
      p q : PNat → Prop
      inst✝¹ : DecidablePred p
      inst✝ : DecidablePred q
      h : Exists fun n => p n
      this : Exists fun n' => Exists fun n => Exists fun x => p n
      n : Subtype fun n => And (Exists fun n_1 => Exists fun x => p n_1) (∀ (m : Nat …
      n' : PNat
      hn' : Eq ↑n ↑n'
      ⊢ LT.lt 0 ↑n
    -/
    rw [hn']
    /-
      case refine_1.intro.intro
      p q : PNat → Prop
      inst✝¹ : DecidablePred p
      inst✝ : DecidablePred q
      h : Exists fun n => p n
      this : Exists fun n' => Exists fun n => Exists fun x => p n
      n : Subtype fun n => And (Exists fun n_1 => Exists fun x => p n_1) (∀ (m : Nat …
      n' : PNat
      hn' : Eq ↑n ↑n'
      ⊢ LT.lt 0 ↑n'
    -/
    exact n'.prop
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      p q : PNat → Prop
      inst✝¹ : DecidablePred p
      inst✝ : DecidablePred q
      h : Exists fun n => p n
      this : Exists fun n' => Exists fun n => Exists fun x => p n
      n : Subtype fun n => And (Exists fun n_1 => Exists fun x => p n_1) (∀ (m : Nat …
      ⊢ p ⟨↑n, ⋯⟩
    -/
  · obtain ⟨n', hn', pn'⟩ := n.prop.1
    /-
      case refine_2.intro.intro
      p q : PNat → Prop
      inst✝¹ : DecidablePred p
      inst✝ : DecidablePred q
      h : Exists fun n => p n
      this : Exists fun n' => Exists fun n => Exists fun x => p n
      n : Subtype fun n => And (Exists fun n_1 => Exists fun x => p n_1) (∀ (m : Nat …
      n' : PNat
      hn' : Eq ↑n ↑n'
      pn' : p n'
      ⊢ p ⟨↑n, ⋯⟩
    -/
    simpa [hn', Subtype.coe_eta] using pn'
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      p q : PNat → Prop
      inst✝¹ : DecidablePred p
      inst✝ : DecidablePred q
      h : Exists fun n => p n
      this : Exists fun n' => Exists fun n => Exists fun x => p n
      n : Subtype fun n => And (Exists fun n_1 => Exists fun x => p n_1) (∀ (m : Nat …
      m : PNat
      hm : LT.lt m ⟨↑n, ⋯⟩
      pm : p m
      ⊢ False
    -/
  · exact n.prop.2 m hm ⟨m, rfl, pm⟩
    /-
      🎉 no goals
    -/


/-- If `p` is a (decidable) predicate on `ℕ+` and `hp : ∃ (n : ℕ+), p n` is a proof that
there exists some positive natural number satisfying `p`, then `PNat.find hp` is the
smallest positive natural number satisfying `p`. Note that `PNat.find` is protected,
meaning that you can't just write `find`, even if the `PNat` namespace is open.

The API for `PNat.find` is:

* `PNat.find_spec` is the proof that `PNat.find hp` satisfies `p`.
* `PNat.find_min` is the proof that if `m < PNat.find hp` then `m` does not satisfy `p`.
* `PNat.find_min'` is the proof that if `m` does satisfy `p` then `PNat.find hp ≤ m`.
-/
protected def find : ℕ+ :=
  PNat.findX h


protected theorem find_spec : p (PNat.find h) :=
  (PNat.findX h).prop.left


protected theorem find_min : ∀ {m : ℕ+}, m < PNat.find h → ¬p m :=
  @(PNat.findX h).prop.right


protected theorem find_min' {m : ℕ+} (hm : p m) : PNat.find h ≤ m :=
  le_of_not_lt fun l => PNat.find_min h l hm


theorem find_eq_iff : PNat.find h = m ↔ p m ∧ ∀ n < m, ¬p n := by
  /-
    p : PNat → Prop
    inst✝ : DecidablePred p
    h : Exists fun n => p n
    m : PNat
    ⊢ Iff (Eq (PNat.find h) m) (And (p m) (∀ (n : PNat), LT.lt n m → Not (p n)))
  -/
  constructor
    /-
      case mp
      p : PNat → Prop
      inst✝ : DecidablePred p
      h : Exists fun n => p n
      m : PNat
      ⊢ Eq (PNat.find h) m → And (p m) (∀ (n : PNat), LT.lt n m → Not (p n))
    -/
  · rintro rfl
    /-
      case mp
      p : PNat → Prop
      inst✝ : DecidablePred p
      h : Exists fun n => p n
      ⊢ And (p (PNat.find h)) (∀ (n : PNat), LT.lt n (PNat.find h) → Not (p n))
    -/
    exact ⟨PNat.find_spec h, fun _ => PNat.find_min h⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      p : PNat → Prop
      inst✝ : DecidablePred p
      h : Exists fun n => p n
      m : PNat
      ⊢ And (p m) (∀ (n : PNat), LT.lt n m → Not (p n)) → Eq (PNat.find h) m
    -/
  · rintro ⟨hm, hlt⟩
    /-
      case mpr.intro
      p : PNat → Prop
      inst✝ : DecidablePred p
      h : Exists fun n => p n
      m : PNat
      hm : p m
      hlt : ∀ (n : PNat), LT.lt n m → Not (p n)
      ⊢ Eq (PNat.find h) m
    -/
    exact le_antisymm (PNat.find_min' h hm) (not_lt.1 <| imp_not_comm.1 (hlt _) <| PNat.find_spec h)
    /-
      🎉 no goals
    -/


@[simp]
theorem find_lt_iff (n : ℕ+) : PNat.find h < n ↔ ∃ m < n, p m :=
  ⟨fun h2 => ⟨PNat.find h, h2, PNat.find_spec h⟩, fun ⟨_, hmn, hm⟩ =>
    (PNat.find_min' h hm).trans_lt hmn⟩


@[simp]
theorem find_le_iff (n : ℕ+) : PNat.find h ≤ n ↔ ∃ m ≤ n, p m := by
  /-
    p : PNat → Prop
    inst✝ : DecidablePred p
    h : Exists fun n => p n
    n : PNat
    ⊢ Iff (LE.le (PNat.find h) n) (Exists fun m => And (LE.le m n) (p m))
  -/
  simp only [exists_prop, ← lt_add_one_iff, find_lt_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem le_find_iff (n : ℕ+) : n ≤ PNat.find h ↔ ∀ m < n, ¬p m := by
  /-
    p : PNat → Prop
    inst✝ : DecidablePred p
    h : Exists fun n => p n
    n : PNat
    ⊢ Iff (LE.le n (PNat.find h)) (∀ (m : PNat), LT.lt m n → Not (p m))
  -/
  simp only [← not_lt, find_lt_iff, not_exists, not_and]
  /-
    🎉 no goals
  -/


@[simp]
theorem lt_find_iff (n : ℕ+) : n < PNat.find h ↔ ∀ m ≤ n, ¬p m := by
  /-
    p : PNat → Prop
    inst✝ : DecidablePred p
    h : Exists fun n => p n
    n : PNat
    ⊢ Iff (LT.lt n (PNat.find h)) (∀ (m : PNat), LE.le m n → Not (p m))
  -/
  simp only [← add_one_le_iff, le_find_iff, add_le_add_iff_right]
  /-
    🎉 no goals
  -/


@[simp]
                                                  /-
                                                    p : PNat → Prop
                                                    inst✝ : DecidablePred p
                                                    h : Exists fun n => p n
                                                    ⊢ Iff (Eq (PNat.find h) 1) (p 1)
                                                  -/
theorem find_eq_one : PNat.find h = 1 ↔ p 1 := by simp [find_eq_iff]
                                                  /-
                                                    🎉 no goals
                                                  -/

-- Porting note: deleted `@[simp]` to satisfy the linter because `le_find_iff` is more general

theorem one_le_find : 1 < PNat.find h ↔ ¬p 1 :=
                       /-
                         p : PNat → Prop
                         inst✝ : DecidablePred p
                         h : Exists fun n => p n
                         ⊢ Iff (Not (LT.lt 1 (PNat.find h))) (Not (Not (p 1)))
                       -/
  not_iff_not.mp <| by simp
                       /-
                         🎉 no goals
                       -/


theorem find_mono (h : ∀ n, q n → p n) {hp : ∃ n, p n} {hq : ∃ n, q n} :
    PNat.find hp ≤ PNat.find hq :=
  PNat.find_min' _ (h _ (PNat.find_spec hq))


theorem find_le {h : ∃ n, p n} (hn : p n) : PNat.find h ≤ n :=
  (PNat.find_le_iff _ _).2 ⟨n, le_rfl, hn⟩


theorem find_comp_succ (h : ∃ n, p n) (h₂ : ∃ n, p (n + 1)) (h1 : ¬p 1) :
    PNat.find h = PNat.find h₂ + 1 := by
  /-
    p : PNat → Prop
    inst✝ : DecidablePred p
    h : Exists fun n => p n
    h₂ : Exists fun n => p (HAdd.hAdd n 1)
    h1 : Not (p 1)
    ⊢ Eq (PNat.find h) (HAdd.hAdd (PNat.find h₂) 1)
  -/
  refine (find_eq_iff _).2 ⟨PNat.find_spec h₂, fun n => PNat.recOn n ?_ ?_⟩
    /-
      case refine_1
      p : PNat → Prop
      inst✝ : DecidablePred p
      h : Exists fun n => p n
      h₂ : Exists fun n => p (HAdd.hAdd n 1)
      h1 : Not (p 1)
      n : PNat
      ⊢ LT.lt 1 (HAdd.hAdd (PNat.find h₂) 1) → Not (p 1)
    -/
  · simp [h1]
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    p : PNat → Prop
    inst✝ : DecidablePred p
    h : Exists fun n => p n
    h₂ : Exists fun n => p (HAdd.hAdd n 1)
    h1 : Not (p 1)
    n : PNat
    ⊢ ∀ (n : PNat), (LT.lt n (HAdd.hAdd (PNat.find h₂) 1) → Not (p n)) → LT.lt (HA …
  -/
  intro m _ hm
  /-
    case refine_2
    p : PNat → Prop
    inst✝ : DecidablePred p
    h : Exists fun n => p n
    h₂ : Exists fun n => p (HAdd.hAdd n 1)
    h1 : Not (p 1)
    n m : PNat
    a✝ : LT.lt m (HAdd.hAdd (PNat.find h₂) 1) → Not (p m)
    hm : LT.lt (HAdd.hAdd m 1) (HAdd.hAdd (PNat.find h₂) 1)
    ⊢ Not (p (HAdd.hAdd m 1))
  -/
  simp only [add_lt_add_iff_right, lt_find_iff] at hm
  /-
    case refine_2
    p : PNat → Prop
    inst✝ : DecidablePred p
    h : Exists fun n => p n
    h₂ : Exists fun n => p (HAdd.hAdd n 1)
    h1 : Not (p 1)
    n m : PNat
    a✝ : LT.lt m (HAdd.hAdd (PNat.find h₂) 1) → Not (p m)
    hm : ∀ (m_1 : PNat), LE.le m_1 m → Not (p (HAdd.hAdd m_1 1))
    ⊢ Not (p (HAdd.hAdd m 1))
  -/
  exact hm _ le_rfl
  /-
    🎉 no goals
  -/


