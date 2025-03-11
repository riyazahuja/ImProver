/-- Effective **Szemerédi Regularity Lemma**: For any sufficiently large graph, there is an
`ε`-uniform equipartition of bounded size (where the bound does not depend on the graph). -/
theorem szemeredi_regularity (hε : 0 < ε) (hl : l ≤ card α) :
    ∃ P : Finpartition univ,
      P.IsEquipartition ∧ l ≤ #P.parts ∧ #P.parts ≤ bound ε l ∧ P.IsUniform G ε := by
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    l : Nat
    hε : LT.lt 0 ε
    hl : LE.le l (Fintype.card α)
    ⊢ Exists fun P => And P.IsEquipartition (And (LE.le l P.parts.card) (And (LE.l …
  -/
  obtain hα | hα := le_total (card α) (bound ε l)
  -- If `card α ≤ bound ε l`, then the partition into singletons is acceptable.
    /-
      case inl
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : Real
      l : Nat
      hε : LT.lt 0 ε
      hl : LE.le l (Fintype.card α)
      hα : LE.le (Fintype.card α) (SzemerediRegularity.bound ε l)
      ⊢ Exists fun P => And P.IsEquipartition (And (LE.le l P.parts.card) (And (LE.l …
    -/
  · refine ⟨⊥, bot_isEquipartition _, ?_⟩
    /-
      case inl
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : Real
      l : Nat
      hε : LT.lt 0 ε
      hl : LE.le l (Fintype.card α)
      hα : LE.le (Fintype.card α) (SzemerediRegularity.bound ε l)
      ⊢ And (LE.le l Bot.bot.parts.card) (And (LE.le Bot.bot.parts.card (SzemerediRe …
    -/
    rw [card_bot, card_univ]
    /-
      case inl
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : Real
      l : Nat
      hε : LT.lt 0 ε
      hl : LE.le l (Fintype.card α)
      hα : LE.le (Fintype.card α) (SzemerediRegularity.bound ε l)
      ⊢ And (LE.le l (Fintype.card α)) (And (LE.le (Fintype.card α) (SzemerediRegula …
    -/
    exact ⟨hl, hα, bot_isUniform _ hε⟩
    /-
      🎉 no goals
    -/
  -- Else, let's start from a dummy equipartition of size `initialBound ε l`.
  /-
    case inr
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    l : Nat
    hε : LT.lt 0 ε
    hl : LE.le l (Fintype.card α)
    hα : LE.le (SzemerediRegularity.bound ε l) (Fintype.card α)
    ⊢ Exists fun P => And P.IsEquipartition (And (LE.le l P.parts.card) (And (LE.l …
  -/
  let t := initialBound ε l
  have htα : t ≤ #(univ : Finset α) :=
    (initialBound_le_bound _ _).trans (by rwa [Finset.card_univ])
  obtain ⟨dum, hdum₁, hdum₂⟩ :=
    exists_equipartition_card_eq (univ : Finset α) (initialBound_pos _ _).ne' htα
  /-
    case inr.intro.intro
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    l : Nat
    hε : LT.lt 0 ε
    hl : LE.le l (Fintype.card α)
    hα : LE.le (SzemerediRegularity.bound ε l) (Fintype.card α)
    t : Nat := SzemerediRegularity.initialBound ε l
    htα : LE.le t Finset.univ.card
    dum : Finpartition Finset.univ
    hdum₁ : dum.IsEquipartition
    hdum₂ : Eq dum.parts.card (SzemerediRegularity.initialBound ε l)
    ⊢ Exists fun P => And P.IsEquipartition (And (LE.le l P.parts.card) (And (LE.l …
  -/
  obtain hε₁ | hε₁ := le_total 1 ε
  -- If `ε ≥ 1`, then this dummy equipartition is `ε`-uniform, so we're done.
  · exact ⟨dum, hdum₁, (le_initialBound ε l).trans hdum₂.ge,
      hdum₂.le.trans (initialBound_le_bound ε l), (dum.isUniform_one G).mono hε₁⟩
  -- Else, set up the induction on energy. We phrase it through the existence for each `i` of an
  -- equipartition of size bounded by `stepBound^[i] (initialBound ε l)` and which is either
  -- `ε`-uniform or has energy at least `ε ^ 5 / 4 * i`.
  have : Nonempty α := by
    rw [← Fintype.card_pos_iff]
    exact (bound_pos _ _).trans_le hα
  suffices h : ∀ i, ∃ P : Finpartition (univ : Finset α), P.IsEquipartition ∧ t ≤ #P.parts ∧
    #P.parts ≤ stepBound^[i] t ∧ (P.IsUniform G ε ∨ ε ^ 5 / 4 * i ≤ P.energy G) by
  -- For `i > 4 / ε ^ 5` we know that the partition we get can't have energy `≥ ε ^ 5 / 4 * i > 1`,
  -- so it must instead be `ε`-uniform and we won.
    obtain ⟨P, hP₁, hP₂, hP₃, hP₄⟩ := h (⌊4 / ε ^ 5⌋₊ + 1)
    refine ⟨P, hP₁, (le_initialBound _ _).trans hP₂, hP₃.trans ?_,
      hP₄.resolve_right fun hPenergy => lt_irrefl (1 : ℝ) ?_⟩
    · rw [iterate_succ_apply', stepBound, bound]
      gcongr
      norm_num
    calc
      (1 : ℝ) = ε ^ 5 / ↑4 * (↑4 / ε ^ 5) := by
        rw [mul_comm, div_mul_div_cancel₀ (pow_pos hε 5).ne']; norm_num
      _ < ε ^ 5 / 4 * (⌊4 / ε ^ 5⌋₊ + 1) :=
        ((mul_lt_mul_left <| by positivity).2 (Nat.lt_floor_add_one _))
      _ ≤ (P.energy G : ℝ) := by rwa [← Nat.cast_add_one]
      _ ≤ 1 := mod_cast P.energy_le_one G
  -- Let's do the actual induction.
  /-
    case inr.intro.intro.inr
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    l : Nat
    hε : LT.lt 0 ε
    hl : LE.le l (Fintype.card α)
    hα : LE.le (SzemerediRegularity.bound ε l) (Fintype.card α)
    t : Nat := SzemerediRegularity.initialBound ε l
    htα : LE.le t Finset.univ.card
    dum : Finpartition Finset.univ
    hdum₁ : dum.IsEquipartition
    hdum₂ : Eq dum.parts.card (SzemerediRegularity.initialBound ε l)
    hε₁ : LE.le ε 1
    this : Nonempty α
    ⊢ ∀ (i : Nat), Exists fun P => And P.IsEquipartition (And (LE.le t P.parts.car …
  -/
  intro i
  /-
    case inr.intro.intro.inr
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    l : Nat
    hε : LT.lt 0 ε
    hl : LE.le l (Fintype.card α)
    hα : LE.le (SzemerediRegularity.bound ε l) (Fintype.card α)
    t : Nat := SzemerediRegularity.initialBound ε l
    htα : LE.le t Finset.univ.card
    dum : Finpartition Finset.univ
    hdum₁ : dum.IsEquipartition
    hdum₂ : Eq dum.parts.card (SzemerediRegularity.initialBound ε l)
    hε₁ : LE.le ε 1
    this : Nonempty α
    i : Nat
    ⊢ Exists fun P => And P.IsEquipartition (And (LE.le t P.parts.card) (And (LE.l …
  -/
  induction' i with i ih
  -- For `i = 0`, the dummy equipartition is enough.
    /-
      case inr.intro.intro.inr.zero
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : Real
      l : Nat
      hε : LT.lt 0 ε
      hl : LE.le l (Fintype.card α)
      hα : LE.le (SzemerediRegularity.bound ε l) (Fintype.card α)
      t : Nat := SzemerediRegularity.initialBound ε l
      htα : LE.le t Finset.univ.card
      dum : Finpartition Finset.univ
      hdum₁ : dum.IsEquipartition
      hdum₂ : Eq dum.parts.card (SzemerediRegularity.initialBound ε l)
      hε₁ : LE.le ε 1
      this : Nonempty α
      ⊢ Exists fun P => And P.IsEquipartition (And (LE.le t P.parts.card) (And (LE.l …
    -/
  · refine ⟨dum, hdum₁, hdum₂.ge, hdum₂.le, Or.inr ?_⟩
    /-
      case inr.intro.intro.inr.zero
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : Real
      l : Nat
      hε : LT.lt 0 ε
      hl : LE.le l (Fintype.card α)
      hα : LE.le (SzemerediRegularity.bound ε l) (Fintype.card α)
      t : Nat := SzemerediRegularity.initialBound ε l
      htα : LE.le t Finset.univ.card
      dum : Finpartition Finset.univ
      hdum₁ : dum.IsEquipartition
      hdum₂ : Eq dum.parts.card (SzemerediRegularity.initialBound ε l)
      hε₁ : LE.le ε 1
      this : Nonempty α
      ⊢ LE.le (HMul.hMul (HDiv.hDiv (HPow.hPow ε 5) 4) ↑0) ↑(dum.energy G)
    -/
    rw [Nat.cast_zero, mul_zero]
    /-
      case inr.intro.intro.inr.zero
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : Real
      l : Nat
      hε : LT.lt 0 ε
      hl : LE.le l (Fintype.card α)
      hα : LE.le (SzemerediRegularity.bound ε l) (Fintype.card α)
      t : Nat := SzemerediRegularity.initialBound ε l
      htα : LE.le t Finset.univ.card
      dum : Finpartition Finset.univ
      hdum₁ : dum.IsEquipartition
      hdum₂ : Eq dum.parts.card (SzemerediRegularity.initialBound ε l)
      hε₁ : LE.le ε 1
      this : Nonempty α
      ⊢ LE.le 0 ↑(dum.energy G)
    -/
    exact mod_cast dum.energy_nonneg G
    /-
      🎉 no goals
    -/
  -- For the induction step at `i + 1`, find `P` the equipartition at `i`.
  /-
    case inr.intro.intro.inr.succ
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    l : Nat
    hε : LT.lt 0 ε
    hl : LE.le l (Fintype.card α)
    hα : LE.le (SzemerediRegularity.bound ε l) (Fintype.card α)
    t : Nat := SzemerediRegularity.initialBound ε l
    htα : LE.le t Finset.univ.card
    dum : Finpartition Finset.univ
    hdum₁ : dum.IsEquipartition
    hdum₂ : Eq dum.parts.card (SzemerediRegularity.initialBound ε l)
    hε₁ : LE.le ε 1
    this : Nonempty α
    i : Nat
    ih : Exists fun P => And P.IsEquipartition (And (LE.le t P.parts.card) (And (L …
    ⊢ Exists fun P => And P.IsEquipartition (And (LE.le t P.parts.card) (And (LE.l …
  -/
  obtain ⟨P, hP₁, hP₂, hP₃, hP₄⟩ := ih
  /-
    case inr.intro.intro.inr.succ.intro.intro.intro.intro
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    l : Nat
    hε : LT.lt 0 ε
    hl : LE.le l (Fintype.card α)
    hα : LE.le (SzemerediRegularity.bound ε l) (Fintype.card α)
    t : Nat := SzemerediRegularity.initialBound ε l
    htα : LE.le t Finset.univ.card
    dum : Finpartition Finset.univ
    hdum₁ : dum.IsEquipartition
    hdum₂ : Eq dum.parts.card (SzemerediRegularity.initialBound ε l)
    hε₁ : LE.le ε 1
    this : Nonempty α
    i : Nat
    P : Finpartition Finset.univ
    hP₁ : P.IsEquipartition
    hP₂ : LE.le t P.parts.card
    hP₃ : LE.le P.parts.card (Nat.iterate SzemerediRegularity.stepBound i t)
    hP₄ : Or (P.IsUniform G ε) (LE.le (HMul.hMul (HDiv.hDiv (HPow.hPow ε 5) 4) ↑i) …
    ⊢ Exists fun P => And P.IsEquipartition (And (LE.le t P.parts.card) (And (LE.l …
  -/
  by_cases huniform : P.IsUniform G ε
  -- If `P` is already uniform, then no need to break it up further. We can just return `P` again.
    /-
      case pos
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : Real
      l : Nat
      hε : LT.lt 0 ε
      hl : LE.le l (Fintype.card α)
      hα : LE.le (SzemerediRegularity.bound ε l) (Fintype.card α)
      t : Nat := SzemerediRegularity.initialBound ε l
      htα : LE.le t Finset.univ.card
      dum : Finpartition Finset.univ
      hdum₁ : dum.IsEquipartition
      hdum₂ : Eq dum.parts.card (SzemerediRegularity.initialBound ε l)
      hε₁ : LE.le ε 1
      this : Nonempty α
      i : Nat
      P : Finpartition Finset.univ
      hP₁ : P.IsEquipartition
      hP₂ : LE.le t P.parts.card
      hP₃ : LE.le P.parts.card (Nat.iterate SzemerediRegularity.stepBound i t)
      hP₄ : Or (P.IsUniform G ε) (LE.le (HMul.hMul (HDiv.hDiv (HPow.hPow ε 5) 4) ↑i) …
      huniform : P.IsUniform G ε
      ⊢ Exists fun P => And P.IsEquipartition (And (LE.le t P.parts.card) (And (LE.l …
    -/
  · refine ⟨P, hP₁, hP₂, ?_, Or.inl huniform⟩
    /-
      case pos
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : Real
      l : Nat
      hε : LT.lt 0 ε
      hl : LE.le l (Fintype.card α)
      hα : LE.le (SzemerediRegularity.bound ε l) (Fintype.card α)
      t : Nat := SzemerediRegularity.initialBound ε l
      htα : LE.le t Finset.univ.card
      dum : Finpartition Finset.univ
      hdum₁ : dum.IsEquipartition
      hdum₂ : Eq dum.parts.card (SzemerediRegularity.initialBound ε l)
      hε₁ : LE.le ε 1
      this : Nonempty α
      i : Nat
      P : Finpartition Finset.univ
      hP₁ : P.IsEquipartition
      hP₂ : LE.le t P.parts.card
      hP₃ : LE.le P.parts.card (Nat.iterate SzemerediRegularity.stepBound i t)
      hP₄ : Or (P.IsUniform G ε) (LE.le (HMul.hMul (HDiv.hDiv (HPow.hPow ε 5) 4) ↑i) …
      huniform : P.IsUniform G ε
      ⊢ LE.le P.parts.card (Nat.iterate SzemerediRegularity.stepBound (HAdd.hAdd i 1 …
    -/
    rw [iterate_succ_apply']
    /-
      case pos
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : Real
      l : Nat
      hε : LT.lt 0 ε
      hl : LE.le l (Fintype.card α)
      hα : LE.le (SzemerediRegularity.bound ε l) (Fintype.card α)
      t : Nat := SzemerediRegularity.initialBound ε l
      htα : LE.le t Finset.univ.card
      dum : Finpartition Finset.univ
      hdum₁ : dum.IsEquipartition
      hdum₂ : Eq dum.parts.card (SzemerediRegularity.initialBound ε l)
      hε₁ : LE.le ε 1
      this : Nonempty α
      i : Nat
      P : Finpartition Finset.univ
      hP₁ : P.IsEquipartition
      hP₂ : LE.le t P.parts.card
      hP₃ : LE.le P.parts.card (Nat.iterate SzemerediRegularity.stepBound i t)
      hP₄ : Or (P.IsUniform G ε) (LE.le (HMul.hMul (HDiv.hDiv (HPow.hPow ε 5) 4) ↑i) …
      huniform : P.IsUniform G ε
      ⊢ LE.le P.parts.card (SzemerediRegularity.stepBound (Nat.iterate SzemerediRegu …
    -/
    exact hP₃.trans (le_stepBound _)
    /-
      🎉 no goals
    -/
  -- Else, `P` must instead have energy at least `ε ^ 5 / 4 * i`.
  /-
    case neg
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    l : Nat
    hε : LT.lt 0 ε
    hl : LE.le l (Fintype.card α)
    hα : LE.le (SzemerediRegularity.bound ε l) (Fintype.card α)
    t : Nat := SzemerediRegularity.initialBound ε l
    htα : LE.le t Finset.univ.card
    dum : Finpartition Finset.univ
    hdum₁ : dum.IsEquipartition
    hdum₂ : Eq dum.parts.card (SzemerediRegularity.initialBound ε l)
    hε₁ : LE.le ε 1
    this : Nonempty α
    i : Nat
    P : Finpartition Finset.univ
    hP₁ : P.IsEquipartition
    hP₂ : LE.le t P.parts.card
    hP₃ : LE.le P.parts.card (Nat.iterate SzemerediRegularity.stepBound i t)
    hP₄ : Or (P.IsUniform G ε) (LE.le (HMul.hMul (HDiv.hDiv (HPow.hPow ε 5) 4) ↑i) …
    huniform : Not (P.IsUniform G ε)
    ⊢ Exists fun P => And P.IsEquipartition (And (LE.le t P.parts.card) (And (LE.l …
  -/
  replace hP₄ := hP₄.resolve_left huniform
  -- We gather a few numerical facts.
  have hεl' : 100 ≤ 4 ^ #P.parts * ε ^ 5 :=
    (hundred_lt_pow_initialBound_mul hε l).le.trans
      (mul_le_mul_of_nonneg_right (pow_right_mono₀ (by norm_num) hP₂) <| by positivity)
  have hi : (i : ℝ) ≤ 4 / ε ^ 5 := by
    have hi : ε ^ 5 / 4 * ↑i ≤ 1 := hP₄.trans (mod_cast P.energy_le_one G)
    rw [div_mul_eq_mul_div, div_le_iff₀ (show (0 : ℝ) < 4 by norm_num)] at hi
    norm_num at hi
    rwa [le_div_iff₀' (pow_pos hε _)]
  have hsize : #P.parts ≤ stepBound^[⌊4 / ε ^ 5⌋₊] t :=
    hP₃.trans (monotone_iterate_of_id_le le_stepBound (Nat.le_floor hi) _)
  have hPα : #P.parts * 16 ^ #P.parts ≤ card α :=
    (Nat.mul_le_mul hsize (Nat.pow_le_pow_of_le_right (by norm_num) hsize)).trans hα
  -- We return the increment equipartition of `P`, which has energy `≥ ε ^ 5 / 4 * (i + 1)`.
  refine ⟨increment hP₁ G ε, increment_isEquipartition hP₁ G ε, ?_, ?_, Or.inr <| le_trans ?_ <|
    energy_increment hP₁ ((seven_le_initialBound ε l).trans hP₂) hεl' hPα huniform hε.le hε₁⟩
    /-
      case neg.refine_1
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : Real
      l : Nat
      hε : LT.lt 0 ε
      hl : LE.le l (Fintype.card α)
      hα : LE.le (SzemerediRegularity.bound ε l) (Fintype.card α)
      t : Nat := SzemerediRegularity.initialBound ε l
      htα : LE.le t Finset.univ.card
      dum : Finpartition Finset.univ
      hdum₁ : dum.IsEquipartition
      hdum₂ : Eq dum.parts.card (SzemerediRegularity.initialBound ε l)
      hε₁ : LE.le ε 1
      this : Nonempty α
      i : Nat
      P : Finpartition Finset.univ
      hP₁ : P.IsEquipartition
      hP₂ : LE.le t P.parts.card
      hP₃ : LE.le P.parts.card (Nat.iterate SzemerediRegularity.stepBound i t)
      huniform : Not (P.IsUniform G ε)
      hP₄ : LE.le (HMul.hMul (HDiv.hDiv (HPow.hPow ε 5) 4) ↑i) ↑(P.energy G)
      hεl' : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
      hi : LE.le (↑i) (HDiv.hDiv 4 (HPow.hPow ε 5))
      hsize : LE.le P.parts.card (Nat.iterate SzemerediRegularity.stepBound (Nat.flo …
      hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
      ⊢ LE.le t (SzemerediRegularity.increment hP₁ G ε).parts.card
    -/
  · rw [card_increment hPα huniform]
    /-
      case neg.refine_1
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : Real
      l : Nat
      hε : LT.lt 0 ε
      hl : LE.le l (Fintype.card α)
      hα : LE.le (SzemerediRegularity.bound ε l) (Fintype.card α)
      t : Nat := SzemerediRegularity.initialBound ε l
      htα : LE.le t Finset.univ.card
      dum : Finpartition Finset.univ
      hdum₁ : dum.IsEquipartition
      hdum₂ : Eq dum.parts.card (SzemerediRegularity.initialBound ε l)
      hε₁ : LE.le ε 1
      this : Nonempty α
      i : Nat
      P : Finpartition Finset.univ
      hP₁ : P.IsEquipartition
      hP₂ : LE.le t P.parts.card
      hP₃ : LE.le P.parts.card (Nat.iterate SzemerediRegularity.stepBound i t)
      huniform : Not (P.IsUniform G ε)
      hP₄ : LE.le (HMul.hMul (HDiv.hDiv (HPow.hPow ε 5) 4) ↑i) ↑(P.energy G)
      hεl' : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
      hi : LE.le (↑i) (HDiv.hDiv 4 (HPow.hPow ε 5))
      hsize : LE.le P.parts.card (Nat.iterate SzemerediRegularity.stepBound (Nat.flo …
      hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
      ⊢ LE.le t (SzemerediRegularity.stepBound P.parts.card)
    -/
    exact hP₂.trans (le_stepBound _)
    /-
      🎉 no goals
    -/
    /-
      case neg.refine_2
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : Real
      l : Nat
      hε : LT.lt 0 ε
      hl : LE.le l (Fintype.card α)
      hα : LE.le (SzemerediRegularity.bound ε l) (Fintype.card α)
      t : Nat := SzemerediRegularity.initialBound ε l
      htα : LE.le t Finset.univ.card
      dum : Finpartition Finset.univ
      hdum₁ : dum.IsEquipartition
      hdum₂ : Eq dum.parts.card (SzemerediRegularity.initialBound ε l)
      hε₁ : LE.le ε 1
      this : Nonempty α
      i : Nat
      P : Finpartition Finset.univ
      hP₁ : P.IsEquipartition
      hP₂ : LE.le t P.parts.card
      hP₃ : LE.le P.parts.card (Nat.iterate SzemerediRegularity.stepBound i t)
      huniform : Not (P.IsUniform G ε)
      hP₄ : LE.le (HMul.hMul (HDiv.hDiv (HPow.hPow ε 5) 4) ↑i) ↑(P.energy G)
      hεl' : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
      hi : LE.le (↑i) (HDiv.hDiv 4 (HPow.hPow ε 5))
      hsize : LE.le P.parts.card (Nat.iterate SzemerediRegularity.stepBound (Nat.flo …
      hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
      ⊢ LE.le (SzemerediRegularity.increment hP₁ G ε).parts.card (Nat.iterate Szemer …
    -/
  · rw [card_increment hPα huniform, iterate_succ_apply']
    /-
      case neg.refine_2
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : Real
      l : Nat
      hε : LT.lt 0 ε
      hl : LE.le l (Fintype.card α)
      hα : LE.le (SzemerediRegularity.bound ε l) (Fintype.card α)
      t : Nat := SzemerediRegularity.initialBound ε l
      htα : LE.le t Finset.univ.card
      dum : Finpartition Finset.univ
      hdum₁ : dum.IsEquipartition
      hdum₂ : Eq dum.parts.card (SzemerediRegularity.initialBound ε l)
      hε₁ : LE.le ε 1
      this : Nonempty α
      i : Nat
      P : Finpartition Finset.univ
      hP₁ : P.IsEquipartition
      hP₂ : LE.le t P.parts.card
      hP₃ : LE.le P.parts.card (Nat.iterate SzemerediRegularity.stepBound i t)
      huniform : Not (P.IsUniform G ε)
      hP₄ : LE.le (HMul.hMul (HDiv.hDiv (HPow.hPow ε 5) 4) ↑i) ↑(P.energy G)
      hεl' : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
      hi : LE.le (↑i) (HDiv.hDiv 4 (HPow.hPow ε 5))
      hsize : LE.le P.parts.card (Nat.iterate SzemerediRegularity.stepBound (Nat.flo …
      hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
      ⊢ LE.le (SzemerediRegularity.stepBound P.parts.card) (SzemerediRegularity.step …
    -/
    exact stepBound_mono hP₃
    /-
      🎉 no goals
    -/
    /-
      case neg.refine_3
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : Real
      l : Nat
      hε : LT.lt 0 ε
      hl : LE.le l (Fintype.card α)
      hα : LE.le (SzemerediRegularity.bound ε l) (Fintype.card α)
      t : Nat := SzemerediRegularity.initialBound ε l
      htα : LE.le t Finset.univ.card
      dum : Finpartition Finset.univ
      hdum₁ : dum.IsEquipartition
      hdum₂ : Eq dum.parts.card (SzemerediRegularity.initialBound ε l)
      hε₁ : LE.le ε 1
      this : Nonempty α
      i : Nat
      P : Finpartition Finset.univ
      hP₁ : P.IsEquipartition
      hP₂ : LE.le t P.parts.card
      hP₃ : LE.le P.parts.card (Nat.iterate SzemerediRegularity.stepBound i t)
      huniform : Not (P.IsUniform G ε)
      hP₄ : LE.le (HMul.hMul (HDiv.hDiv (HPow.hPow ε 5) 4) ↑i) ↑(P.energy G)
      hεl' : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
      hi : LE.le (↑i) (HDiv.hDiv 4 (HPow.hPow ε 5))
      hsize : LE.le P.parts.card (Nat.iterate SzemerediRegularity.stepBound (Nat.flo …
      hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
      ⊢ LE.le (HMul.hMul (HDiv.hDiv (HPow.hPow ε 5) 4) ↑(HAdd.hAdd i 1)) (HAdd.hAdd  …
    -/
  · rw [Nat.cast_succ, mul_add, mul_one]
    /-
      case neg.refine_3
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : Real
      l : Nat
      hε : LT.lt 0 ε
      hl : LE.le l (Fintype.card α)
      hα : LE.le (SzemerediRegularity.bound ε l) (Fintype.card α)
      t : Nat := SzemerediRegularity.initialBound ε l
      htα : LE.le t Finset.univ.card
      dum : Finpartition Finset.univ
      hdum₁ : dum.IsEquipartition
      hdum₂ : Eq dum.parts.card (SzemerediRegularity.initialBound ε l)
      hε₁ : LE.le ε 1
      this : Nonempty α
      i : Nat
      P : Finpartition Finset.univ
      hP₁ : P.IsEquipartition
      hP₂ : LE.le t P.parts.card
      hP₃ : LE.le P.parts.card (Nat.iterate SzemerediRegularity.stepBound i t)
      huniform : Not (P.IsUniform G ε)
      hP₄ : LE.le (HMul.hMul (HDiv.hDiv (HPow.hPow ε 5) 4) ↑i) ↑(P.energy G)
      hεl' : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
      hi : LE.le (↑i) (HDiv.hDiv 4 (HPow.hPow ε 5))
      hsize : LE.le P.parts.card (Nat.iterate SzemerediRegularity.stepBound (Nat.flo …
      hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
      ⊢ LE.le (HAdd.hAdd (HMul.hMul (HDiv.hDiv (HPow.hPow ε 5) 4) ↑i) (HDiv.hDiv (HP …
    -/
    exact add_le_add_right hP₄ _
    /-
      🎉 no goals
    -/

