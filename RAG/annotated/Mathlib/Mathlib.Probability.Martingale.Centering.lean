/-- Any `ℕ`-indexed stochastic process can be written as the sum of a martingale and a predictable
process. This is the predictable process. See `martingalePart` for the martingale. -/
noncomputable def predictablePart {m0 : MeasurableSpace Ω} (f : ℕ → Ω → E) (ℱ : Filtration ℕ m0)
    (μ : Measure Ω) : ℕ → Ω → E := fun n => ∑ i ∈ Finset.range n, μ[f (i + 1) - f i|ℱ i]


@[simp]
theorem predictablePart_zero : predictablePart f ℱ μ 0 = 0 := by
  /-
    Ω : Type u_1
    E : Type u_2
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f : Nat → Ω → E
    ℱ : MeasureTheory.Filtration Nat m0
    ⊢ Eq (MeasureTheory.predictablePart f ℱ μ 0) 0
  -/
  simp_rw [predictablePart, Finset.range_zero, Finset.sum_empty]
  /-
    🎉 no goals
  -/


theorem adapted_predictablePart : Adapted ℱ fun n => predictablePart f ℱ μ (n + 1) := fun _ =>
  Finset.stronglyMeasurable_sum' _ fun _ hin =>
    stronglyMeasurable_condexp.mono (ℱ.mono (Finset.mem_range_succ_iff.mp hin))


theorem adapted_predictablePart' : Adapted ℱ fun n => predictablePart f ℱ μ n := fun _ =>
  Finset.stronglyMeasurable_sum' _ fun _ hin =>
    stronglyMeasurable_condexp.mono (ℱ.mono (Finset.mem_range_le hin))


/-- Any `ℕ`-indexed stochastic process can be written as the sum of a martingale and a predictable
process. This is the martingale. See `predictablePart` for the predictable process. -/
noncomputable def martingalePart {m0 : MeasurableSpace Ω} (f : ℕ → Ω → E) (ℱ : Filtration ℕ m0)
    (μ : Measure Ω) : ℕ → Ω → E := fun n => f n - predictablePart f ℱ μ n


theorem martingalePart_add_predictablePart (ℱ : Filtration ℕ m0) (μ : Measure Ω) (f : ℕ → Ω → E) :
    martingalePart f ℱ μ + predictablePart f ℱ μ = f :=
  sub_add_cancel _ _


theorem martingalePart_eq_sum : martingalePart f ℱ μ = fun n =>
    f 0 + ∑ i ∈ Finset.range n, (f (i + 1) - f i - μ[f (i + 1) - f i|ℱ i]) := by
  /-
    Ω : Type u_1
    E : Type u_2
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f : Nat → Ω → E
    ℱ : MeasureTheory.Filtration Nat m0
    ⊢ Eq (MeasureTheory.martingalePart f ℱ μ) fun n => HAdd.hAdd (f 0) ((Finset.ra …
  -/
  unfold martingalePart predictablePart
  /-
    Ω : Type u_1
    E : Type u_2
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f : Nat → Ω → E
    ℱ : MeasureTheory.Filtration Nat m0
    ⊢ Eq (fun n => HSub.hSub (f n) ((Finset.range n).sum fun i => MeasureTheory.co …
  -/
  ext1 n
  /-
    case h
    Ω : Type u_1
    E : Type u_2
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f : Nat → Ω → E
    ℱ : MeasureTheory.Filtration Nat m0
    n : Nat
    ⊢ Eq (HSub.hSub (f n) ((Finset.range n).sum fun i => MeasureTheory.condexp (↑ℱ …
  -/
  rw [Finset.eq_sum_range_sub f n, ← add_sub, ← Finset.sum_sub_distrib]
  /-
    🎉 no goals
  -/


theorem adapted_martingalePart (hf : Adapted ℱ f) : Adapted ℱ (martingalePart f ℱ μ) :=
  Adapted.sub hf adapted_predictablePart'


theorem integrable_martingalePart (hf_int : ∀ n, Integrable (f n) μ) (n : ℕ) :
    Integrable (martingalePart f ℱ μ n) μ := by
  /-
    Ω : Type u_1
    E : Type u_2
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f : Nat → Ω → E
    ℱ : MeasureTheory.Filtration Nat m0
    hf_int : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    n : Nat
    ⊢ MeasureTheory.Integrable (MeasureTheory.martingalePart f ℱ μ n) μ
  -/
  rw [martingalePart_eq_sum]
  exact (hf_int 0).add
    (integrable_finset_sum' _ fun i _ => ((hf_int _).sub (hf_int _)).sub integrable_condexp)


theorem martingale_martingalePart (hf : Adapted ℱ f) (hf_int : ∀ n, Integrable (f n) μ)
    [SigmaFiniteFiltration μ ℱ] : Martingale (martingalePart f ℱ μ) ℱ μ := by
  /-
    Ω : Type u_1
    E : Type u_2
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    f : Nat → Ω → E
    ℱ : MeasureTheory.Filtration Nat m0
    hf : MeasureTheory.Adapted ℱ f
    hf_int : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    ⊢ MeasureTheory.Martingale (MeasureTheory.martingalePart f ℱ μ) ℱ μ
  -/
  refine ⟨adapted_martingalePart hf, fun i j hij => ?_⟩
  -- ⊢ μ[martingalePart f ℱ μ j | ℱ i] =ᵐ[μ] martingalePart f ℱ μ i
  have h_eq_sum : μ[martingalePart f ℱ μ j|ℱ i] =ᵐ[μ]
      f 0 + ∑ k ∈ Finset.range j, (μ[f (k + 1) - f k|ℱ i] - μ[μ[f (k + 1) - f k|ℱ k]|ℱ i]) := by
    rw [martingalePart_eq_sum]
    refine (condexp_add (hf_int 0) ?_).trans ?_
    · exact integrable_finset_sum' _ fun i _ => ((hf_int _).sub (hf_int _)).sub integrable_condexp
    refine (EventuallyEq.add EventuallyEq.rfl (condexp_finset_sum fun i _ => ?_)).trans ?_
    · exact ((hf_int _).sub (hf_int _)).sub integrable_condexp
    refine EventuallyEq.add ?_ ?_
    · rw [condexp_of_stronglyMeasurable (ℱ.le _) _ (hf_int 0)]
      · exact (hf 0).mono (ℱ.mono (zero_le i))
    · exact eventuallyEq_sum fun k _ => condexp_sub ((hf_int _).sub (hf_int _)) integrable_condexp
  /-
    Ω : Type u_1
    E : Type u_2
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    f : Nat → Ω → E
    ℱ : MeasureTheory.Filtration Nat m0
    hf : MeasureTheory.Adapted ℱ f
    hf_int : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    i j : Nat
    hij : LE.le i j
    h_eq_sum : (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp (↑ℱ i) μ ( …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp (↑ℱ i) μ (MeasureTh …
  -/
  refine h_eq_sum.trans ?_
  have h_ge : ∀ k, i ≤ k → μ[f (k + 1) - f k|ℱ i] - μ[μ[f (k + 1) - f k|ℱ k]|ℱ i] =ᵐ[μ] 0 := by
    intro k hk
    have : μ[μ[f (k + 1) - f k|ℱ k]|ℱ i] =ᵐ[μ] μ[f (k + 1) - f k|ℱ i] :=
      condexp_condexp_of_le (ℱ.mono hk) (ℱ.le k)
    filter_upwards [this] with x hx
    rw [Pi.sub_apply, Pi.zero_apply, hx, sub_self]
  have h_lt : ∀ k, k < i → μ[f (k + 1) - f k|ℱ i] - μ[μ[f (k + 1) - f k|ℱ k]|ℱ i] =ᵐ[μ]
      f (k + 1) - f k - μ[f (k + 1) - f k|ℱ k] := by
    refine fun k hk => EventuallyEq.sub ?_ ?_
    · rw [condexp_of_stronglyMeasurable]
      · exact ((hf (k + 1)).mono (ℱ.mono (Nat.succ_le_of_lt hk))).sub ((hf k).mono (ℱ.mono hk.le))
      · exact (hf_int _).sub (hf_int _)
    · rw [condexp_of_stronglyMeasurable]
      · exact stronglyMeasurable_condexp.mono (ℱ.mono hk.le)
      · exact integrable_condexp
  /-
    Ω : Type u_1
    E : Type u_2
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    f : Nat → Ω → E
    ℱ : MeasureTheory.Filtration Nat m0
    hf : MeasureTheory.Adapted ℱ f
    hf_int : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    i j : Nat
    hij : LE.le i j
    h_eq_sum : (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp (↑ℱ i) μ ( …
    h_ge : ∀ (k : Nat), LE.le i k → (MeasureTheory.ae μ).EventuallyEq (HSub.hSub ( …
    h_lt : ∀ (k : Nat), LT.lt k i → (MeasureTheory.ae μ).EventuallyEq (HSub.hSub ( …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (HAdd.hAdd (f 0) ((Finset.range j).sum fun …
  -/
  rw [martingalePart_eq_sum]
  /-
    Ω : Type u_1
    E : Type u_2
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    f : Nat → Ω → E
    ℱ : MeasureTheory.Filtration Nat m0
    hf : MeasureTheory.Adapted ℱ f
    hf_int : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    i j : Nat
    hij : LE.le i j
    h_eq_sum : (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp (↑ℱ i) μ ( …
    h_ge : ∀ (k : Nat), LE.le i k → (MeasureTheory.ae μ).EventuallyEq (HSub.hSub ( …
    h_lt : ∀ (k : Nat), LT.lt k i → (MeasureTheory.ae μ).EventuallyEq (HSub.hSub ( …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (HAdd.hAdd (f 0) ((Finset.range j).sum fun …
  -/
  refine EventuallyEq.add EventuallyEq.rfl ?_
  rw [← Finset.sum_range_add_sum_Ico _ hij, ←
    add_zero (∑ i ∈ Finset.range i, (f (i + 1) - f i - μ[f (i + 1) - f i|ℱ i]))]
  /-
    Ω : Type u_1
    E : Type u_2
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    f : Nat → Ω → E
    ℱ : MeasureTheory.Filtration Nat m0
    hf : MeasureTheory.Adapted ℱ f
    hf_int : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    i j : Nat
    hij : LE.le i j
    h_eq_sum : (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp (↑ℱ i) μ ( …
    h_ge : ∀ (k : Nat), LE.le i k → (MeasureTheory.ae μ).EventuallyEq (HSub.hSub ( …
    h_lt : ∀ (k : Nat), LT.lt k i → (MeasureTheory.ae μ).EventuallyEq (HSub.hSub ( …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (HAdd.hAdd ((Finset.range i).sum fun k =>  …
  -/
  refine (eventuallyEq_sum fun k hk => h_lt k (Finset.mem_range.mp hk)).add ?_
  /-
    Ω : Type u_1
    E : Type u_2
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    f : Nat → Ω → E
    ℱ : MeasureTheory.Filtration Nat m0
    hf : MeasureTheory.Adapted ℱ f
    hf_int : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    i j : Nat
    hij : LE.le i j
    h_eq_sum : (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp (↑ℱ i) μ ( …
    h_ge : ∀ (k : Nat), LE.le i k → (MeasureTheory.ae μ).EventuallyEq (HSub.hSub ( …
    h_lt : ∀ (k : Nat), LT.lt k i → (MeasureTheory.ae μ).EventuallyEq (HSub.hSub ( …
    ⊢ (MeasureTheory.ae μ).EventuallyEq ((Finset.Ico i j).sum fun k => HSub.hSub ( …
  -/
  refine (eventuallyEq_sum fun k hk => h_ge k (Finset.mem_Ico.mp hk).1).trans ?_
  /-
    Ω : Type u_1
    E : Type u_2
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    f : Nat → Ω → E
    ℱ : MeasureTheory.Filtration Nat m0
    hf : MeasureTheory.Adapted ℱ f
    hf_int : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    i j : Nat
    hij : LE.le i j
    h_eq_sum : (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp (↑ℱ i) μ ( …
    h_ge : ∀ (k : Nat), LE.le i k → (MeasureTheory.ae μ).EventuallyEq (HSub.hSub ( …
    h_lt : ∀ (k : Nat), LT.lt k i → (MeasureTheory.ae μ).EventuallyEq (HSub.hSub ( …
    ⊢ (MeasureTheory.ae μ).EventuallyEq ((Finset.Ico i j).sum fun i => 0) 0
  -/
  simp only [Finset.sum_const_zero, Pi.zero_apply]
  /-
    Ω : Type u_1
    E : Type u_2
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    f : Nat → Ω → E
    ℱ : MeasureTheory.Filtration Nat m0
    hf : MeasureTheory.Adapted ℱ f
    hf_int : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    i j : Nat
    hij : LE.le i j
    h_eq_sum : (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp (↑ℱ i) μ ( …
    h_ge : ∀ (k : Nat), LE.le i k → (MeasureTheory.ae μ).EventuallyEq (HSub.hSub ( …
    h_lt : ∀ (k : Nat), LT.lt k i → (MeasureTheory.ae μ).EventuallyEq (HSub.hSub ( …
    ⊢ (MeasureTheory.ae μ).EventuallyEq 0 0
  -/
  rfl
  /-
    🎉 no goals
  -/

-- The following two lemmas demonstrate the essential uniqueness of the decomposition

theorem martingalePart_add_ae_eq [SigmaFiniteFiltration μ ℱ] {f g : ℕ → Ω → E}
    (hf : Martingale f ℱ μ) (hg : Adapted ℱ fun n => g (n + 1)) (hg0 : g 0 = 0)
    (hgint : ∀ n, Integrable (g n) μ) (n : ℕ) : martingalePart (f + g) ℱ μ n =ᵐ[μ] f n := by
  /-
    Ω : Type u_1
    E : Type u_2
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    ℱ : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    f g : Nat → Ω → E
    hf : MeasureTheory.Martingale f ℱ μ
    hg : MeasureTheory.Adapted ℱ fun n => g (HAdd.hAdd n 1)
    hg0 : Eq (g 0) 0
    hgint : ∀ (n : Nat), MeasureTheory.Integrable (g n) μ
    n : Nat
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.martingalePart (HAdd.hAdd f …
  -/
  set h := f - martingalePart (f + g) ℱ μ with hhdef
  have hh : h = predictablePart (f + g) ℱ μ - g := by
    rw [hhdef, sub_eq_sub_iff_add_eq_add, add_comm (predictablePart (f + g) ℱ μ),
      martingalePart_add_predictablePart]
  have hhpred : Adapted ℱ fun n => h (n + 1) := by
    rw [hh]
    exact adapted_predictablePart.sub hg
  have hhmgle : Martingale h ℱ μ := hf.sub (martingale_martingalePart
    (hf.adapted.add <| Predictable.adapted hg <| hg0.symm ▸ stronglyMeasurable_zero) fun n =>
    (hf.integrable n).add <| hgint n)
  /-
    Ω : Type u_1
    E : Type u_2
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    ℱ : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    f g : Nat → Ω → E
    hf : MeasureTheory.Martingale f ℱ μ
    hg : MeasureTheory.Adapted ℱ fun n => g (HAdd.hAdd n 1)
    hg0 : Eq (g 0) 0
    hgint : ∀ (n : Nat), MeasureTheory.Integrable (g n) μ
    n : Nat
    h : Nat → Ω → E := HSub.hSub f (MeasureTheory.martingalePart (HAdd.hAdd f g) ℱ …
    hhdef : Eq h (HSub.hSub f (MeasureTheory.martingalePart (HAdd.hAdd f g) ℱ μ))
    hh : Eq h (HSub.hSub (MeasureTheory.predictablePart (HAdd.hAdd f g) ℱ μ) g)
    hhpred : MeasureTheory.Adapted ℱ fun n => h (HAdd.hAdd n 1)
    hhmgle : MeasureTheory.Martingale h ℱ μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.martingalePart (HAdd.hAdd f …
  -/
  refine (eventuallyEq_iff_sub.2 ?_).symm
  /-
    Ω : Type u_1
    E : Type u_2
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    ℱ : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    f g : Nat → Ω → E
    hf : MeasureTheory.Martingale f ℱ μ
    hg : MeasureTheory.Adapted ℱ fun n => g (HAdd.hAdd n 1)
    hg0 : Eq (g 0) 0
    hgint : ∀ (n : Nat), MeasureTheory.Integrable (g n) μ
    n : Nat
    h : Nat → Ω → E := HSub.hSub f (MeasureTheory.martingalePart (HAdd.hAdd f g) ℱ …
    hhdef : Eq h (HSub.hSub f (MeasureTheory.martingalePart (HAdd.hAdd f g) ℱ μ))
    hh : Eq h (HSub.hSub (MeasureTheory.predictablePart (HAdd.hAdd f g) ℱ μ) g)
    hhpred : MeasureTheory.Adapted ℱ fun n => h (HAdd.hAdd n 1)
    hhmgle : MeasureTheory.Martingale h ℱ μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (HSub.hSub (f n) (MeasureTheory.martingale …
  -/
  filter_upwards [hhmgle.eq_zero_of_predictable hhpred n] with ω hω
  /-
    case h
    Ω : Type u_1
    E : Type u_2
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    ℱ : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    f g : Nat → Ω → E
    hf : MeasureTheory.Martingale f ℱ μ
    hg : MeasureTheory.Adapted ℱ fun n => g (HAdd.hAdd n 1)
    hg0 : Eq (g 0) 0
    hgint : ∀ (n : Nat), MeasureTheory.Integrable (g n) μ
    n : Nat
    h : Nat → Ω → E := HSub.hSub f (MeasureTheory.martingalePart (HAdd.hAdd f g) ℱ …
    hhdef : Eq h (HSub.hSub f (MeasureTheory.martingalePart (HAdd.hAdd f g) ℱ μ))
    hh : Eq h (HSub.hSub (MeasureTheory.predictablePart (HAdd.hAdd f g) ℱ μ) g)
    hhpred : MeasureTheory.Adapted ℱ fun n => h (HAdd.hAdd n 1)
    hhmgle : MeasureTheory.Martingale h ℱ μ
    ω : Ω
    hω : Eq (h n ω) (h 0 ω)
    ⊢ Eq (HSub.hSub (f n) (MeasureTheory.martingalePart (HAdd.hAdd f g) ℱ μ n) ω)  …
  -/
  unfold h at hω
  /-
    case h
    Ω : Type u_1
    E : Type u_2
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    ℱ : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    f g : Nat → Ω → E
    hf : MeasureTheory.Martingale f ℱ μ
    hg : MeasureTheory.Adapted ℱ fun n => g (HAdd.hAdd n 1)
    hg0 : Eq (g 0) 0
    hgint : ∀ (n : Nat), MeasureTheory.Integrable (g n) μ
    n : Nat
    h : Nat → Ω → E := HSub.hSub f (MeasureTheory.martingalePart (HAdd.hAdd f g) ℱ …
    hhdef : Eq h (HSub.hSub f (MeasureTheory.martingalePart (HAdd.hAdd f g) ℱ μ))
    hh : Eq h (HSub.hSub (MeasureTheory.predictablePart (HAdd.hAdd f g) ℱ μ) g)
    hhpred : MeasureTheory.Adapted ℱ fun n => h (HAdd.hAdd n 1)
    hhmgle : MeasureTheory.Martingale h ℱ μ
    ω : Ω
    hω : Eq (HSub.hSub f (MeasureTheory.martingalePart (HAdd.hAdd f g) ℱ μ) n ω) ( …
    ⊢ Eq (HSub.hSub (f n) (MeasureTheory.martingalePart (HAdd.hAdd f g) ℱ μ n) ω)  …
  -/
  rw [Pi.sub_apply] at hω
  /-
    case h
    Ω : Type u_1
    E : Type u_2
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    ℱ : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    f g : Nat → Ω → E
    hf : MeasureTheory.Martingale f ℱ μ
    hg : MeasureTheory.Adapted ℱ fun n => g (HAdd.hAdd n 1)
    hg0 : Eq (g 0) 0
    hgint : ∀ (n : Nat), MeasureTheory.Integrable (g n) μ
    n : Nat
    h : Nat → Ω → E := HSub.hSub f (MeasureTheory.martingalePart (HAdd.hAdd f g) ℱ …
    hhdef : Eq h (HSub.hSub f (MeasureTheory.martingalePart (HAdd.hAdd f g) ℱ μ))
    hh : Eq h (HSub.hSub (MeasureTheory.predictablePart (HAdd.hAdd f g) ℱ μ) g)
    hhpred : MeasureTheory.Adapted ℱ fun n => h (HAdd.hAdd n 1)
    hhmgle : MeasureTheory.Martingale h ℱ μ
    ω : Ω
    hω : Eq (HSub.hSub (f n) (MeasureTheory.martingalePart (HAdd.hAdd f g) ℱ μ n)  …
    ⊢ Eq (HSub.hSub (f n) (MeasureTheory.martingalePart (HAdd.hAdd f g) ℱ μ n) ω)  …
  -/
  rw [hω, Pi.sub_apply, martingalePart]
  /-
    case h
    Ω : Type u_1
    E : Type u_2
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    ℱ : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    f g : Nat → Ω → E
    hf : MeasureTheory.Martingale f ℱ μ
    hg : MeasureTheory.Adapted ℱ fun n => g (HAdd.hAdd n 1)
    hg0 : Eq (g 0) 0
    hgint : ∀ (n : Nat), MeasureTheory.Integrable (g n) μ
    n : Nat
    h : Nat → Ω → E := HSub.hSub f (MeasureTheory.martingalePart (HAdd.hAdd f g) ℱ …
    hhdef : Eq h (HSub.hSub f (MeasureTheory.martingalePart (HAdd.hAdd f g) ℱ μ))
    hh : Eq h (HSub.hSub (MeasureTheory.predictablePart (HAdd.hAdd f g) ℱ μ) g)
    hhpred : MeasureTheory.Adapted ℱ fun n => h (HAdd.hAdd n 1)
    hhmgle : MeasureTheory.Martingale h ℱ μ
    ω : Ω
    hω : Eq (HSub.hSub (f n) (MeasureTheory.martingalePart (HAdd.hAdd f g) ℱ μ n)  …
    ⊢ Eq (HSub.hSub (f 0) (HSub.hSub (HAdd.hAdd f g 0) (MeasureTheory.predictableP …
  -/
  simp [hg0]
  /-
    🎉 no goals
  -/


theorem predictablePart_add_ae_eq [SigmaFiniteFiltration μ ℱ] {f g : ℕ → Ω → E}
    (hf : Martingale f ℱ μ) (hg : Adapted ℱ fun n => g (n + 1)) (hg0 : g 0 = 0)
    (hgint : ∀ n, Integrable (g n) μ) (n : ℕ) : predictablePart (f + g) ℱ μ n =ᵐ[μ] g n := by
  /-
    Ω : Type u_1
    E : Type u_2
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    ℱ : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    f g : Nat → Ω → E
    hf : MeasureTheory.Martingale f ℱ μ
    hg : MeasureTheory.Adapted ℱ fun n => g (HAdd.hAdd n 1)
    hg0 : Eq (g 0) 0
    hgint : ∀ (n : Nat), MeasureTheory.Integrable (g n) μ
    n : Nat
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.predictablePart (HAdd.hAdd  …
  -/
  filter_upwards [martingalePart_add_ae_eq hf hg hg0 hgint n] with ω hω
  /-
    case h
    Ω : Type u_1
    E : Type u_2
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    ℱ : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    f g : Nat → Ω → E
    hf : MeasureTheory.Martingale f ℱ μ
    hg : MeasureTheory.Adapted ℱ fun n => g (HAdd.hAdd n 1)
    hg0 : Eq (g 0) 0
    hgint : ∀ (n : Nat), MeasureTheory.Integrable (g n) μ
    n : Nat
    ω : Ω
    hω : Eq (MeasureTheory.martingalePart (HAdd.hAdd f g) ℱ μ n ω) (f n ω)
    ⊢ Eq (MeasureTheory.predictablePart (HAdd.hAdd f g) ℱ μ n ω) (g n ω)
  -/
  rw [← add_right_inj (f n ω)]
  /-
    case h
    Ω : Type u_1
    E : Type u_2
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    ℱ : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    f g : Nat → Ω → E
    hf : MeasureTheory.Martingale f ℱ μ
    hg : MeasureTheory.Adapted ℱ fun n => g (HAdd.hAdd n 1)
    hg0 : Eq (g 0) 0
    hgint : ∀ (n : Nat), MeasureTheory.Integrable (g n) μ
    n : Nat
    ω : Ω
    hω : Eq (MeasureTheory.martingalePart (HAdd.hAdd f g) ℱ μ n ω) (f n ω)
    ⊢ Eq (HAdd.hAdd (f n ω) (MeasureTheory.predictablePart (HAdd.hAdd f g) ℱ μ n ω …
  -/
  conv_rhs => rw [← Pi.add_apply, ← Pi.add_apply, ← martingalePart_add_predictablePart ℱ μ (f + g)]
  /-
    case h
    Ω : Type u_1
    E : Type u_2
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    ℱ : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    f g : Nat → Ω → E
    hf : MeasureTheory.Martingale f ℱ μ
    hg : MeasureTheory.Adapted ℱ fun n => g (HAdd.hAdd n 1)
    hg0 : Eq (g 0) 0
    hgint : ∀ (n : Nat), MeasureTheory.Integrable (g n) μ
    n : Nat
    ω : Ω
    hω : Eq (MeasureTheory.martingalePart (HAdd.hAdd f g) ℱ μ n ω) (f n ω)
    ⊢ Eq (HAdd.hAdd (f n ω) (MeasureTheory.predictablePart (HAdd.hAdd f g) ℱ μ n ω …
  -/
  rw [Pi.add_apply, Pi.add_apply, hω]
  /-
    🎉 no goals
  -/


theorem predictablePart_bdd_difference {R : ℝ≥0} {f : ℕ → Ω → ℝ} (ℱ : Filtration ℕ m0)
    (hbdd : ∀ᵐ ω ∂μ, ∀ i, |f (i + 1) ω - f i ω| ≤ R) :
    ∀ᵐ ω ∂μ, ∀ i, |predictablePart f ℱ μ (i + 1) ω - predictablePart f ℱ μ i ω| ≤ R := by
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    R : NNReal
    f : Nat → Ω → Real
    ℱ : MeasureTheory.Filtration Nat m0
    hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
    ⊢ Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (MeasureTheor …
  -/
  simp_rw [predictablePart, Finset.sum_apply, Finset.sum_range_succ_sub_sum]
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    R : NNReal
    f : Nat → Ω → Real
    ℱ : MeasureTheory.Filtration Nat m0
    hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
    ⊢ Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (MeasureTheory.condexp ( …
  -/
  exact ae_all_iff.2 fun i => ae_bdd_condexp_of_ae_bdd <| ae_all_iff.1 hbdd i
  /-
    🎉 no goals
  -/


theorem martingalePart_bdd_difference {R : ℝ≥0} {f : ℕ → Ω → ℝ} (ℱ : Filtration ℕ m0)
    (hbdd : ∀ᵐ ω ∂μ, ∀ i, |f (i + 1) ω - f i ω| ≤ R) :
    ∀ᵐ ω ∂μ, ∀ i, |martingalePart f ℱ μ (i + 1) ω - martingalePart f ℱ μ i ω| ≤ ↑(2 * R) := by
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    R : NNReal
    f : Nat → Ω → Real
    ℱ : MeasureTheory.Filtration Nat m0
    hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
    ⊢ Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (MeasureTheor …
  -/
  filter_upwards [hbdd, predictablePart_bdd_difference ℱ hbdd] with ω hω₁ hω₂ i
  /-
    case h
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    R : NNReal
    f : Nat → Ω → Real
    ℱ : MeasureTheory.Filtration Nat m0
    hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
    ω : Ω
    hω₁ : ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd.hAdd i 1) ω) (f i ω))) ↑R
    hω₂ : ∀ (i : Nat), LE.le (abs (HSub.hSub (MeasureTheory.predictablePart f ℱ μ  …
    i : Nat
    ⊢ LE.le (abs (HSub.hSub (MeasureTheory.martingalePart f ℱ μ (HAdd.hAdd i 1) ω) …
  -/
  simp only [two_mul, martingalePart, Pi.sub_apply]
  have : |f (i + 1) ω - predictablePart f ℱ μ (i + 1) ω - (f i ω - predictablePart f ℱ μ i ω)| =
      |f (i + 1) ω - f i ω - (predictablePart f ℱ μ (i + 1) ω - predictablePart f ℱ μ i ω)| := by
    ring_nf -- `ring` suggests `ring_nf` despite proving the goal
  /-
    case h
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    R : NNReal
    f : Nat → Ω → Real
    ℱ : MeasureTheory.Filtration Nat m0
    hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
    ω : Ω
    hω₁ : ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd.hAdd i 1) ω) (f i ω))) ↑R
    hω₂ : ∀ (i : Nat), LE.le (abs (HSub.hSub (MeasureTheory.predictablePart f ℱ μ  …
    i : Nat
    this : Eq (abs (HSub.hSub (HSub.hSub (f (HAdd.hAdd i 1) ω) (MeasureTheory.pred …
    ⊢ LE.le (abs (HSub.hSub (HSub.hSub (f (HAdd.hAdd i 1) ω) (MeasureTheory.predic …
  -/
  rw [this]
  /-
    case h
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    R : NNReal
    f : Nat → Ω → Real
    ℱ : MeasureTheory.Filtration Nat m0
    hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
    ω : Ω
    hω₁ : ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd.hAdd i 1) ω) (f i ω))) ↑R
    hω₂ : ∀ (i : Nat), LE.le (abs (HSub.hSub (MeasureTheory.predictablePart f ℱ μ  …
    i : Nat
    this : Eq (abs (HSub.hSub (HSub.hSub (f (HAdd.hAdd i 1) ω) (MeasureTheory.pred …
    ⊢ LE.le (abs (HSub.hSub (HSub.hSub (f (HAdd.hAdd i 1) ω) (f i ω)) (HSub.hSub ( …
  -/
  exact (abs_sub _ _).trans (add_le_add (hω₁ i) (hω₂ i))
  /-
    🎉 no goals
  -/


