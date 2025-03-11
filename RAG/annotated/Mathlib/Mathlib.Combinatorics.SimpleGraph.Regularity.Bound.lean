/-- Auxiliary function for Szemerédi's regularity lemma. Blowing up a partition of size `n` during
the induction results in a partition of size at most `stepBound n`. -/
def stepBound (n : ℕ) : ℕ :=
  n * 4 ^ n


theorem le_stepBound : id ≤ stepBound := fun n =>
                                           /-
                                             n : Nat
                                             ⊢ LT.lt 0 4
                                           -/
  Nat.le_mul_of_pos_right _ <| pow_pos (by norm_num) n
                                           /-
                                             🎉 no goals
                                           -/


theorem stepBound_mono : Monotone stepBound := fun _ _ h =>
                                                     /-
                                                       x✝¹ x✝ : Nat
                                                       h : LE.le x✝¹ x✝
                                                       ⊢ GT.gt 4 0
                                                     -/
  Nat.mul_le_mul h <| Nat.pow_le_pow_of_le_right (by norm_num) h
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem stepBound_pos_iff {n : ℕ} : 0 < stepBound n ↔ 0 < n :=
                                 /-
                                   n : Nat
                                   ⊢ LT.lt 0 (HPow.hPow 4 n)
                                 -/
  mul_pos_iff_of_pos_right <| by positivity
                                 /-
                                   🎉 no goals
                                 -/


alias ⟨_, stepBound_pos⟩ := stepBound_pos_iff


@[norm_cast] lemma coe_stepBound {α : Type*} [Semiring α] (n : ℕ) :
                                        /-
                                          α : Type u_1
                                          inst✝ : Semiring α
                                          n : Nat
                                          ⊢ Eq (↑(SzemerediRegularity.stepBound n)) (HMul.hMul (↑n) (HPow.hPow 4 n))
                                        -/
    (stepBound n : α) = n * 4 ^ n := by unfold stepBound; norm_cast
                                                          /-
                                                            🎉 no goals
                                                          -/


local notation3 "m" => (card α / stepBound #P.parts : ℕ)


local notation3 "a" => (card α / #P.parts - m * 4 ^ #P.parts : ℕ)


private theorem eps_pos {ε : ℝ} {n : ℕ} (h : 100 ≤ (4 : ℝ) ^ n * ε ^ 5) : 0 < ε :=
                       /-
                         ε : Real
                         n : Nat
                         h : LE.le 100 (HMul.hMul (HPow.hPow 4 n) (HPow.hPow ε 5))
                         ⊢ Odd 5
                       -/
  (Odd.pow_pos_iff (by decide)).mp
                       /-
                         🎉 no goals
                       -/
                                                  /-
                                                    ε : Real
                                                    n : Nat
                                                    h : LE.le 100 (HMul.hMul (HPow.hPow 4 n) (HPow.hPow ε 5))
                                                    ⊢ LT.lt 0 100
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
    (pos_of_mul_pos_right ((show 0 < (100 : ℝ) by norm_num).trans_le h) (by positivity))
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


private theorem m_pos [Nonempty α] (hPα : #P.parts * 16 ^ #P.parts ≤ card α) : 0 < m :=
                                /-
                                  α : Type u_1
                                  inst✝² : DecidableEq α
                                  inst✝¹ : Fintype α
                                  P : Finpartition Finset.univ
                                  inst✝ : Nonempty α
                                  hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
                                  ⊢ LE.le (SzemerediRegularity.stepBound P.parts.card) (HMul.hMul P.parts.card ( …
                                -/
  Nat.div_pos (hPα.trans' <| by unfold stepBound; gcongr; norm_num) <|
                                                          /-
                                                            🎉 no goals
                                                          -/
    stepBound_pos (P.parts_nonempty <| univ_nonempty.ne_empty).card_pos


/-- Local extension for the `positivity` tactic: A few facts that are needed many times for the
proof of Szemerédi's regularity lemma. -/
-- Porting note: positivity extensions must now be global, and this did not seem like a good
-- match for positivity anymore, so I wrote a new tactic (kmill)
scoped macro "sz_positivity" : tactic =>
  `(tactic|
      { try have := m_pos ‹_›
        try have := eps_pos ‹_›
        positivity })

-- Original meta code
/- meta def positivity_szemeredi_regularity : expr → tactic strictness
| `(%%n / step_bound (finpartition.parts %%P).card) := do
    p ← to_expr
      ``((finpartition.parts %%P).card * 16^(finpartition.parts %%P).card ≤ %%n)
      >>= find_assumption,
    positive <$> mk_app ``m_pos [p]
| ε := do
    typ ← infer_type ε,
    unify typ `(ℝ),
    p ← to_expr ``(100 ≤ 4 ^ _ * %%ε ^ 5) >>= find_assumption,
    positive <$> mk_app ``eps_pos [p] -/


theorem m_pos [Nonempty α] (hPα : #P.parts * 16 ^ #P.parts ≤ card α) : 0 < m := by
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    P : Finpartition Finset.univ
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    ⊢ LT.lt 0 (HDiv.hDiv (Fintype.card α) (SzemerediRegularity.stepBound P.parts.c …
  -/
  sz_positivity
  /-
    🎉 no goals
  -/


                                                  /-
                                                    α : Type u_1
                                                    inst✝¹ : DecidableEq α
                                                    inst✝ : Fintype α
                                                    P : Finpartition Finset.univ
                                                    ⊢ LT.lt 0 (HAdd.hAdd (↑(HDiv.hDiv (Fintype.card α) (SzemerediRegularity.stepBo …
                                                  -/
theorem coe_m_add_one_pos : 0 < (m : ℝ) + 1 := by positivity
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem one_le_m_coe [Nonempty α] (hPα : #P.parts * 16 ^ #P.parts ≤ card α) : (1 : ℝ) ≤ m :=
  Nat.one_le_cast.2 <| m_pos hPα


theorem eps_pow_five_pos (hPε : 100 ≤ (4 : ℝ) ^ #P.parts * ε ^ 5) : ↑0 < ε ^ 5 :=
                            /-
                              α : Type u_1
                              inst✝¹ : DecidableEq α
                              inst✝ : Fintype α
                              P : Finpartition Finset.univ
                              ε : Real
                              hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
                              ⊢ LT.lt 0 100
                            -/
                            /-
                              🎉 no goals
                            -/
  pos_of_mul_pos_right ((by norm_num : (0 : ℝ) < 100).trans_le hPε) <| pow_nonneg (by norm_num) _
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


theorem eps_pos (hPε : 100 ≤ (4 : ℝ) ^ #P.parts * ε ^ 5) : 0 < ε :=
                       /-
                         α : Type u_1
                         inst✝¹ : DecidableEq α
                         inst✝ : Fintype α
                         P : Finpartition Finset.univ
                         ε : Real
                         hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
                         ⊢ Odd 5
                       -/
  (Odd.pow_pos_iff (by decide)).mp (eps_pow_five_pos hPε)
                       /-
                         🎉 no goals
                       -/


theorem hundred_div_ε_pow_five_le_m [Nonempty α] (hPα : #P.parts * 16 ^ #P.parts ≤ card α)
    (hPε : 100 ≤ (4 : ℝ) ^ #P.parts * ε ^ 5) : 100 / ε ^ 5 ≤ m :=
                                                   /-
                                                     α : Type u_1
                                                     inst✝² : DecidableEq α
                                                     inst✝¹ : Fintype α
                                                     P : Finpartition Finset.univ
                                                     ε : Real
                                                     inst✝ : Nonempty α
                                                     hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
                                                     hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
                                                     ⊢ LE.le 0 (HPow.hPow 4 P.parts.card)
                                                   -/
  (div_le_of_le_mul₀ (eps_pow_five_pos hPε).le (by positivity) hPε).trans <| by
                                                   /-
                                                     🎉 no goals
                                                   -/
    /-
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      P : Finpartition Finset.univ
      ε : Real
      inst✝ : Nonempty α
      hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
      hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
      ⊢ LE.le (HPow.hPow 4 P.parts.card) ↑(HDiv.hDiv (Fintype.card α) (SzemerediRegu …
    -/
    norm_cast
    rwa [Nat.le_div_iff_mul_le (stepBound_pos (P.parts_nonempty <|
      univ_nonempty.ne_empty).card_pos), stepBound, mul_left_comm, ← mul_pow]


theorem hundred_le_m [Nonempty α] (hPα : #P.parts * 16 ^ #P.parts ≤ card α)
    (hPε : 100 ≤ (4 : ℝ) ^ #P.parts * ε ^ 5) (hε : ε ≤ 1) : 100 ≤ m :=
  mod_cast
    (hundred_div_ε_pow_five_le_m hPα hPε).trans'
                       /-
                         α : Type u_1
                         inst✝² : DecidableEq α
                         inst✝¹ : Fintype α
                         P : Finpartition Finset.univ
                         ε : Real
                         inst✝ : Nonempty α
                         hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
                         hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
                         hε : LE.le ε 1
                         ⊢ LE.le 0 100
                       -/
                       /-
                         🎉 no goals
                       -/
                                     /-
                                       🎉 no goals
                                     -/
      (le_div_self (by norm_num) (by sz_positivity) <| pow_le_one₀ (by sz_positivity) hε)
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem a_add_one_le_four_pow_parts_card : a + 1 ≤ 4 ^ #P.parts := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    P : Finpartition Finset.univ
    ⊢ LE.le (HAdd.hAdd (HSub.hSub (HDiv.hDiv (Fintype.card α) P.parts.card) (HMul. …
  -/
  have h : 1 ≤ 4 ^ #P.parts := one_le_pow₀ (by norm_num)
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    P : Finpartition Finset.univ
    h : LE.le 1 (HPow.hPow 4 P.parts.card)
    ⊢ LE.le (HAdd.hAdd (HSub.hSub (HDiv.hDiv (Fintype.card α) P.parts.card) (HMul. …
  -/
  rw [stepBound, ← Nat.div_div_eq_div_mul]
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    P : Finpartition Finset.univ
    h : LE.le 1 (HPow.hPow 4 P.parts.card)
    ⊢ LE.le (HAdd.hAdd (HSub.hSub (HDiv.hDiv (Fintype.card α) P.parts.card) (HMul. …
  -/
  conv_rhs => rw [← Nat.sub_add_cancel h]
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    P : Finpartition Finset.univ
    h : LE.le 1 (HPow.hPow 4 P.parts.card)
    ⊢ LE.le (HAdd.hAdd (HSub.hSub (HDiv.hDiv (Fintype.card α) P.parts.card) (HMul. …
  -/
  rw [add_le_add_iff_right, tsub_le_iff_left, ← Nat.add_sub_assoc h]
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    P : Finpartition Finset.univ
    h : LE.le 1 (HPow.hPow 4 P.parts.card)
    ⊢ LE.le (HDiv.hDiv (Fintype.card α) P.parts.card) (HSub.hSub (HAdd.hAdd (HMul. …
  -/
  exact Nat.le_sub_one_of_lt (Nat.lt_div_mul_add h)
  /-
    🎉 no goals
  -/


theorem card_aux₁ (hucard : #u = m * 4 ^ #P.parts + a) :
    (4 ^ #P.parts - a) * m + a * (m + 1) = #u := by
  rw [hucard, mul_add, mul_one, ← add_assoc, ← add_mul,
    Nat.sub_add_cancel ((Nat.le_succ _).trans a_add_one_le_four_pow_parts_card), mul_comm]


theorem card_aux₂ (hP : P.IsEquipartition) (hu : u ∈ P.parts) (hucard : #u ≠ m * 4 ^ #P.parts + a) :
    (4 ^ #P.parts - (a + 1)) * m + (a + 1) * (m + 1) = #u := by
  have : m * 4 ^ #P.parts ≤ card α / #P.parts := by
    rw [stepBound, ← Nat.div_div_eq_div_mul]
    exact Nat.div_mul_le_self _ _
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    P : Finpartition Finset.univ
    u : Finset α
    hP : P.IsEquipartition
    hu : Membership.mem P.parts u
    hucard : Ne u.card (HAdd.hAdd (HMul.hMul (HDiv.hDiv (Fintype.card α) (Szemered …
    this : LE.le (HMul.hMul (HDiv.hDiv (Fintype.card α) (SzemerediRegularity.stepB …
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HSub.hSub (HPow.hPow 4 P.parts.card) (HAdd.hAdd (H …
  -/
  rw [Nat.add_sub_of_le this] at hucard
  rw [(hP.card_parts_eq_average hu).resolve_left hucard, mul_add, mul_one, ← add_assoc, ← add_mul,
    Nat.sub_add_cancel a_add_one_le_four_pow_parts_card, ← add_assoc, mul_comm,
    Nat.add_sub_of_le this, card_univ]


theorem pow_mul_m_le_card_part (hP : P.IsEquipartition) (hu : u ∈ P.parts) :
    (4 : ℝ) ^ #P.parts * m ≤ #u := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    P : Finpartition Finset.univ
    u : Finset α
    hP : P.IsEquipartition
    hu : Membership.mem P.parts u
    ⊢ LE.le (HMul.hMul (HPow.hPow 4 P.parts.card) ↑(HDiv.hDiv (Fintype.card α) (Sz …
  -/
  norm_cast
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    P : Finpartition Finset.univ
    u : Finset α
    hP : P.IsEquipartition
    hu : Membership.mem P.parts u
    ⊢ LE.le (HMul.hMul (HPow.hPow 4 P.parts.card) (HDiv.hDiv (Fintype.card α) (Sze …
  -/
  rw [stepBound, ← Nat.div_div_eq_div_mul]
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    P : Finpartition Finset.univ
    u : Finset α
    hP : P.IsEquipartition
    hu : Membership.mem P.parts u
    ⊢ LE.le (HMul.hMul (HPow.hPow 4 P.parts.card) (HDiv.hDiv (HDiv.hDiv (Fintype.c …
  -/
  exact (Nat.mul_div_le _ _).trans (hP.average_le_card_part hu)
  /-
    🎉 no goals
  -/


/-- Auxiliary function for Szemerédi's regularity lemma. The size of the partition by which we start
blowing. -/
noncomputable def initialBound : ℕ :=
  max 7 <| max l <| ⌊log (100 / ε ^ 5) / log 4⌋₊ + 1


theorem le_initialBound : l ≤ initialBound ε l :=
  (le_max_left _ _).trans <| le_max_right _ _


theorem seven_le_initialBound : 7 ≤ initialBound ε l :=
  le_max_left _ _


theorem initialBound_pos : 0 < initialBound ε l :=
  Nat.succ_pos'.trans_le <| seven_le_initialBound _ _


theorem hundred_lt_pow_initialBound_mul {ε : ℝ} (hε : 0 < ε) (l : ℕ) :
    100 < ↑4 ^ initialBound ε l * ε ^ 5 := by
  rw [← rpow_natCast 4, ← div_lt_iff₀ (pow_pos hε 5), lt_rpow_iff_log_lt _ zero_lt_four, ←
    div_lt_iff₀, initialBound, Nat.cast_max, Nat.cast_max]
    /-
      ε : Real
      hε : LT.lt 0 ε
      l : Nat
      ⊢ LT.lt (HDiv.hDiv (Real.log (HDiv.hDiv 100 (HPow.hPow ε 5))) (Real.log 4)) (M …
    -/
  · push_cast
    /-
      ε : Real
      hε : LT.lt 0 ε
      l : Nat
      ⊢ LT.lt (HDiv.hDiv (Real.log (HDiv.hDiv 100 (HPow.hPow ε 5))) (Real.log 4)) (M …
    -/
    exact lt_max_of_lt_right (lt_max_of_lt_right <| Nat.lt_floor_add_one _)
    /-
      🎉 no goals
    -/
    /-
      ε : Real
      hε : LT.lt 0 ε
      l : Nat
      ⊢ LT.lt 0 (Real.log 4)
    -/
  · exact log_pos (by norm_num)
    /-
      🎉 no goals
    -/
    /-
      ε : Real
      hε : LT.lt 0 ε
      l : Nat
      ⊢ LT.lt 0 (HDiv.hDiv 100 (HPow.hPow ε 5))
    -/
  · exact div_pos (by norm_num) (pow_pos hε 5)
    /-
      🎉 no goals
    -/


/-- An explicit bound on the size of the equipartition whose existence is given by Szemerédi's
regularity lemma. -/
noncomputable def bound : ℕ :=
  (stepBound^[⌊4 / ε ^ 5⌋₊] <| initialBound ε l) *
    16 ^ (stepBound^[⌊4 / ε ^ 5⌋₊] <| initialBound ε l)


theorem initialBound_le_bound : initialBound ε l ≤ bound ε l :=
                                                                                     /-
                                                                                       ε : Real
                                                                                       l : Nat
                                                                                       ⊢ LT.lt 0 (HPow.hPow 16 (Nat.iterate SzemerediRegularity.stepBound (Nat.floor  …
                                                                                     -/
  (id_le_iterate_of_id_le le_stepBound _ _).trans <| Nat.le_mul_of_pos_right _ <| by positivity
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


theorem le_bound : l ≤ bound ε l :=
  (le_initialBound ε l).trans <| initialBound_le_bound ε l


theorem bound_pos : 0 < bound ε l :=
  (initialBound_pos ε l).trans_le <| initialBound_le_bound ε l


theorem mul_sq_le_sum_sq (hst : s ⊆ t) (f : ι → 𝕜) (hs : x ^ 2 ≤ ((∑ i ∈ s, f i) / #s) ^ 2)
    (hs' : (#s : 𝕜) ≠ 0) : (#s : 𝕜) * x ^ 2 ≤ ∑ i ∈ t, f i ^ 2 :=
  (mul_le_mul_of_nonneg_left (hs.trans sum_div_card_sq_le_sum_sq_div_card) <|
    Nat.cast_nonneg _).trans <| (mul_div_cancel₀ _ hs').le.trans <|
      sum_le_sum_of_subset_of_nonneg hst fun _ _ _ => sq_nonneg _


theorem add_div_le_sum_sq_div_card (hst : s ⊆ t) (f : ι → 𝕜) (d : 𝕜) (hx : 0 ≤ x)
    (hs : x ≤ |(∑ i ∈ s, f i) / #s - (∑ i ∈ t, f i) / #t|) (ht : d ≤ ((∑ i ∈ t, f i) / #t) ^ 2) :
    d + #s / #t * x ^ 2 ≤ (∑ i ∈ t, f i ^ 2) / #t := by
  /-
    ι : Type u_2
    𝕜 : Type u_3
    inst✝ : LinearOrderedField 𝕜
    s t : Finset ι
    x : 𝕜
    hst : HasSubset.Subset s t
    f : ι → 𝕜
    d : 𝕜
    hx : LE.le 0 x
    hs : LE.le x (abs (HSub.hSub (HDiv.hDiv (s.sum fun i => f i) ↑s.card) (HDiv.hD …
    ht : LE.le d (HPow.hPow (HDiv.hDiv (t.sum fun i => f i) ↑t.card) 2)
    ⊢ LE.le (HAdd.hAdd d (HMul.hMul (HDiv.hDiv ↑s.card ↑t.card) (HPow.hPow x 2)))  …
  -/
  obtain hscard | hscard := ((#s).cast_nonneg : (0 : 𝕜) ≤ #s).eq_or_lt
    /-
      case inl
      ι : Type u_2
      𝕜 : Type u_3
      inst✝ : LinearOrderedField 𝕜
      s t : Finset ι
      x : 𝕜
      hst : HasSubset.Subset s t
      f : ι → 𝕜
      d : 𝕜
      hx : LE.le 0 x
      hs : LE.le x (abs (HSub.hSub (HDiv.hDiv (s.sum fun i => f i) ↑s.card) (HDiv.hD …
      ht : LE.le d (HPow.hPow (HDiv.hDiv (t.sum fun i => f i) ↑t.card) 2)
      hscard : Eq 0 ↑s.card
      ⊢ LE.le (HAdd.hAdd d (HMul.hMul (HDiv.hDiv ↑s.card ↑t.card) (HPow.hPow x 2)))  …
    -/
  · simpa [← hscard] using ht.trans sum_div_card_sq_le_sum_sq_div_card
    /-
      🎉 no goals
    -/
  /-
    case inr
    ι : Type u_2
    𝕜 : Type u_3
    inst✝ : LinearOrderedField 𝕜
    s t : Finset ι
    x : 𝕜
    hst : HasSubset.Subset s t
    f : ι → 𝕜
    d : 𝕜
    hx : LE.le 0 x
    hs : LE.le x (abs (HSub.hSub (HDiv.hDiv (s.sum fun i => f i) ↑s.card) (HDiv.hD …
    ht : LE.le d (HPow.hPow (HDiv.hDiv (t.sum fun i => f i) ↑t.card) 2)
    hscard : LT.lt 0 ↑s.card
    ⊢ LE.le (HAdd.hAdd d (HMul.hMul (HDiv.hDiv ↑s.card ↑t.card) (HPow.hPow x 2)))  …
  -/
  have htcard : (0 : 𝕜) < #t := hscard.trans_le (Nat.cast_le.2 (card_le_card hst))
  have h₁ : x ^ 2 ≤ ((∑ i ∈ s, f i) / #s - (∑ i ∈ t, f i) / #t) ^ 2 :=
    sq_le_sq.2 (by rwa [abs_of_nonneg hx])
  have h₂ : x ^ 2 ≤ ((∑ i ∈ s, (f i - (∑ j ∈ t, f j) / #t)) / #s) ^ 2 := by
    apply h₁.trans
    rw [sum_sub_distrib, sum_const, nsmul_eq_mul, sub_div, mul_div_cancel_left₀ _ hscard.ne']
  /-
    case inr
    ι : Type u_2
    𝕜 : Type u_3
    inst✝ : LinearOrderedField 𝕜
    s t : Finset ι
    x : 𝕜
    hst : HasSubset.Subset s t
    f : ι → 𝕜
    d : 𝕜
    hx : LE.le 0 x
    hs : LE.le x (abs (HSub.hSub (HDiv.hDiv (s.sum fun i => f i) ↑s.card) (HDiv.hD …
    ht : LE.le d (HPow.hPow (HDiv.hDiv (t.sum fun i => f i) ↑t.card) 2)
    hscard : LT.lt 0 ↑s.card
    htcard : LT.lt 0 ↑t.card
    h₁ : LE.le (HPow.hPow x 2) (HPow.hPow (HSub.hSub (HDiv.hDiv (s.sum fun i => f  …
    h₂ : LE.le (HPow.hPow x 2) (HPow.hPow (HDiv.hDiv (s.sum fun i => HSub.hSub (f  …
    ⊢ LE.le (HAdd.hAdd d (HMul.hMul (HDiv.hDiv ↑s.card ↑t.card) (HPow.hPow x 2)))  …
  -/
  apply (add_le_add_right ht _).trans
  /-
    case inr
    ι : Type u_2
    𝕜 : Type u_3
    inst✝ : LinearOrderedField 𝕜
    s t : Finset ι
    x : 𝕜
    hst : HasSubset.Subset s t
    f : ι → 𝕜
    d : 𝕜
    hx : LE.le 0 x
    hs : LE.le x (abs (HSub.hSub (HDiv.hDiv (s.sum fun i => f i) ↑s.card) (HDiv.hD …
    ht : LE.le d (HPow.hPow (HDiv.hDiv (t.sum fun i => f i) ↑t.card) 2)
    hscard : LT.lt 0 ↑s.card
    htcard : LT.lt 0 ↑t.card
    h₁ : LE.le (HPow.hPow x 2) (HPow.hPow (HSub.hSub (HDiv.hDiv (s.sum fun i => f  …
    h₂ : LE.le (HPow.hPow x 2) (HPow.hPow (HDiv.hDiv (s.sum fun i => HSub.hSub (f  …
    ⊢ LE.le (HAdd.hAdd (HPow.hPow (HDiv.hDiv (t.sum fun i => f i) ↑t.card) 2) (HMu …
  -/
  rw [← mul_div_right_comm, le_div_iff₀ htcard, add_mul, div_mul_cancel₀ _ htcard.ne']
  /-
    case inr
    ι : Type u_2
    𝕜 : Type u_3
    inst✝ : LinearOrderedField 𝕜
    s t : Finset ι
    x : 𝕜
    hst : HasSubset.Subset s t
    f : ι → 𝕜
    d : 𝕜
    hx : LE.le 0 x
    hs : LE.le x (abs (HSub.hSub (HDiv.hDiv (s.sum fun i => f i) ↑s.card) (HDiv.hD …
    ht : LE.le d (HPow.hPow (HDiv.hDiv (t.sum fun i => f i) ↑t.card) 2)
    hscard : LT.lt 0 ↑s.card
    htcard : LT.lt 0 ↑t.card
    h₁ : LE.le (HPow.hPow x 2) (HPow.hPow (HSub.hSub (HDiv.hDiv (s.sum fun i => f  …
    h₂ : LE.le (HPow.hPow x 2) (HPow.hPow (HDiv.hDiv (s.sum fun i => HSub.hSub (f  …
    ⊢ LE.le (HAdd.hAdd (HMul.hMul (HPow.hPow (HDiv.hDiv (t.sum fun i => f i) ↑t.ca …
  -/
  have h₃ := mul_sq_le_sum_sq hst (fun i => (f i - (∑ j ∈ t, f j) / #t)) h₂ hscard.ne'
  /-
    case inr
    ι : Type u_2
    𝕜 : Type u_3
    inst✝ : LinearOrderedField 𝕜
    s t : Finset ι
    x : 𝕜
    hst : HasSubset.Subset s t
    f : ι → 𝕜
    d : 𝕜
    hx : LE.le 0 x
    hs : LE.le x (abs (HSub.hSub (HDiv.hDiv (s.sum fun i => f i) ↑s.card) (HDiv.hD …
    ht : LE.le d (HPow.hPow (HDiv.hDiv (t.sum fun i => f i) ↑t.card) 2)
    hscard : LT.lt 0 ↑s.card
    htcard : LT.lt 0 ↑t.card
    h₁ : LE.le (HPow.hPow x 2) (HPow.hPow (HSub.hSub (HDiv.hDiv (s.sum fun i => f  …
    h₂ : LE.le (HPow.hPow x 2) (HPow.hPow (HDiv.hDiv (s.sum fun i => HSub.hSub (f  …
    h₃ : LE.le (HMul.hMul (↑s.card) (HPow.hPow x 2)) (t.sum fun i => HPow.hPow ((f …
    ⊢ LE.le (HAdd.hAdd (HMul.hMul (HPow.hPow (HDiv.hDiv (t.sum fun i => f i) ↑t.ca …
  -/
  apply (add_le_add_left h₃ _).trans
  -- Porting note: was
  -- `simp [← mul_div_right_comm _ (#t : 𝕜), sub_div' _ _ _ htcard.ne', ← sum_div, ← add_div,`
  -- `  mul_pow, div_le_iff₀ (sq_pos_of_ne_zero htcard.ne'), sub_sq, sum_add_distrib, ← sum_mul,`
  -- `  ← mul_sum]`
  /-
    case inr
    ι : Type u_2
    𝕜 : Type u_3
    inst✝ : LinearOrderedField 𝕜
    s t : Finset ι
    x : 𝕜
    hst : HasSubset.Subset s t
    f : ι → 𝕜
    d : 𝕜
    hx : LE.le 0 x
    hs : LE.le x (abs (HSub.hSub (HDiv.hDiv (s.sum fun i => f i) ↑s.card) (HDiv.hD …
    ht : LE.le d (HPow.hPow (HDiv.hDiv (t.sum fun i => f i) ↑t.card) 2)
    hscard : LT.lt 0 ↑s.card
    htcard : LT.lt 0 ↑t.card
    h₁ : LE.le (HPow.hPow x 2) (HPow.hPow (HSub.hSub (HDiv.hDiv (s.sum fun i => f  …
    h₂ : LE.le (HPow.hPow x 2) (HPow.hPow (HDiv.hDiv (s.sum fun i => HSub.hSub (f  …
    h₃ : LE.le (HMul.hMul (↑s.card) (HPow.hPow x 2)) (t.sum fun i => HPow.hPow ((f …
    ⊢ LE.le (HAdd.hAdd (HMul.hMul (HPow.hPow (HDiv.hDiv (t.sum fun i => f i) ↑t.ca …
  -/
  simp_rw [sub_div' _ _ _ htcard.ne']
  /-
    case inr
    ι : Type u_2
    𝕜 : Type u_3
    inst✝ : LinearOrderedField 𝕜
    s t : Finset ι
    x : 𝕜
    hst : HasSubset.Subset s t
    f : ι → 𝕜
    d : 𝕜
    hx : LE.le 0 x
    hs : LE.le x (abs (HSub.hSub (HDiv.hDiv (s.sum fun i => f i) ↑s.card) (HDiv.hD …
    ht : LE.le d (HPow.hPow (HDiv.hDiv (t.sum fun i => f i) ↑t.card) 2)
    hscard : LT.lt 0 ↑s.card
    htcard : LT.lt 0 ↑t.card
    h₁ : LE.le (HPow.hPow x 2) (HPow.hPow (HSub.hSub (HDiv.hDiv (s.sum fun i => f  …
    h₂ : LE.le (HPow.hPow x 2) (HPow.hPow (HDiv.hDiv (s.sum fun i => HSub.hSub (f  …
    h₃ : LE.le (HMul.hMul (↑s.card) (HPow.hPow x 2)) (t.sum fun i => HPow.hPow ((f …
    ⊢ LE.le (HAdd.hAdd (HMul.hMul (HPow.hPow (HDiv.hDiv (t.sum fun i => f i) ↑t.ca …
  -/
  conv_lhs => enter [2, 2, x]; rw [div_pow]
  rw [div_pow, ← sum_div, ← mul_div_right_comm _ (#t : 𝕜), ← add_div,
    div_le_iff₀ (sq_pos_of_ne_zero htcard.ne')]
  simp_rw [sub_sq, sum_add_distrib, sum_const, nsmul_eq_mul, sum_sub_distrib, mul_pow, ← sum_mul,
    ← mul_sum, ← sum_mul]
  /-
    case inr
    ι : Type u_2
    𝕜 : Type u_3
    inst✝ : LinearOrderedField 𝕜
    s t : Finset ι
    x : 𝕜
    hst : HasSubset.Subset s t
    f : ι → 𝕜
    d : 𝕜
    hx : LE.le 0 x
    hs : LE.le x (abs (HSub.hSub (HDiv.hDiv (s.sum fun i => f i) ↑s.card) (HDiv.hD …
    ht : LE.le d (HPow.hPow (HDiv.hDiv (t.sum fun i => f i) ↑t.card) 2)
    hscard : LT.lt 0 ↑s.card
    htcard : LT.lt 0 ↑t.card
    h₁ : LE.le (HPow.hPow x 2) (HPow.hPow (HSub.hSub (HDiv.hDiv (s.sum fun i => f  …
    h₂ : LE.le (HPow.hPow x 2) (HPow.hPow (HDiv.hDiv (s.sum fun i => HSub.hSub (f  …
    h₃ : LE.le (HMul.hMul (↑s.card) (HPow.hPow x 2)) (t.sum fun i => HPow.hPow ((f …
    ⊢ LE.le (HAdd.hAdd (HMul.hMul (HPow.hPow (t.sum fun i => f i) 2) ↑t.card) (HAd …
  -/
  ring_nf; rfl
           /-
             🎉 no goals
           -/


/-- Extension for the `positivity` tactic: `SzemerediRegularity.initialBound` is always positive. -/
@[positivity SzemerediRegularity.initialBound _ _]
def evalInitialBound : PositivityExt where eval {u α} _ _ e := do
  match u, α, e with
  | 0, ~q(ℕ), ~q(SzemerediRegularity.initialBound $ε $l) =>
    assertInstancesCommute
    pure (.positive q(SzemerediRegularity.initialBound_pos $ε $l))
  | _, _, _ => throwError "not initialBound"



/-- Extension for the `positivity` tactic: `SzemerediRegularity.bound` is always positive. -/
@[positivity SzemerediRegularity.bound _ _]
def evalBound : PositivityExt where eval {u α} _ _ e := do
  match u, α, e with
  | 0, ~q(ℕ), ~q(SzemerediRegularity.bound $ε $l) =>
    assertInstancesCommute
    pure (.positive q(SzemerediRegularity.bound_pos $ε $l))
  | _, _, _ => throwError "not bound"


