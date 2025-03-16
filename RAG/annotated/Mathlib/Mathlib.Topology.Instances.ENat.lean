/--
Topology on `ℕ∞`.

Note: this is different from the `EMetricSpace` topology. The `EMetricSpace` topology has
`IsOpen {∞}`, but all neighborhoods of `∞` in `ℕ∞` contain infinite intervals.
-/
instance : TopologicalSpace ℕ∞ := Preorder.topology ℕ∞


instance : OrderTopology ℕ∞ := ⟨rfl⟩


@[simp] theorem range_natCast : range ((↑) : ℕ → ℕ∞) = Iio ⊤ :=
  WithTop.range_coe


theorem isEmbedding_natCast : IsEmbedding ((↑) : ℕ → ℕ∞) :=
  Nat.strictMono_cast.isEmbedding_of_ordConnected <| range_natCast ▸ ordConnected_Iio


@[deprecated (since := "2024-10-26")]
alias embedding_natCast := isEmbedding_natCast


theorem isOpenEmbedding_natCast : IsOpenEmbedding ((↑) : ℕ → ℕ∞) :=
  ⟨isEmbedding_natCast, range_natCast ▸ isOpen_Iio⟩


@[deprecated (since := "2024-10-18")]
alias openEmbedding_natCast := isOpenEmbedding_natCast


theorem nhds_natCast (n : ℕ) : 𝓝 (n : ℕ∞) = pure (n : ℕ∞) := by
  /-
    n : Nat
    ⊢ Eq (nhds ↑n) (Pure.pure ↑n)
  -/
  simp [← isOpenEmbedding_natCast.map_nhds_eq]
  /-
    🎉 no goals
  -/


@[simp]
protected theorem nhds_eq_pure {n : ℕ∞} (h : n ≠ ⊤) : 𝓝 n = pure n := by
  /-
    n : ENat
    h : Ne n Top.top
    ⊢ Eq (nhds n) (Pure.pure n)
  -/
  lift n to ℕ using h
  /-
    case intro
    n : Nat
    ⊢ Eq (nhds ↑n) (Pure.pure ↑n)
  -/
  simp [nhds_natCast]
  /-
    🎉 no goals
  -/


theorem isOpen_singleton {x : ℕ∞} (hx : x ≠ ⊤) : IsOpen {x} := by
  /-
    x : ENat
    hx : Ne x Top.top
    ⊢ IsOpen (Singleton.singleton x)
  -/
  rw [isOpen_singleton_iff_nhds_eq_pure, ENat.nhds_eq_pure hx]
  /-
    🎉 no goals
  -/


theorem mem_nhds_iff {x : ℕ∞} {s : Set ℕ∞} (hx : x ≠ ⊤) : s ∈ 𝓝 x ↔ x ∈ s := by
  /-
    x : ENat
    s : Set ENat
    hx : Ne x Top.top
    ⊢ Iff (Membership.mem (nhds x) s) (Membership.mem s x)
  -/
  simp [hx]
  /-
    🎉 no goals
  -/


theorem mem_nhds_natCast_iff (n : ℕ) {s : Set ℕ∞} : s ∈ 𝓝 (n : ℕ∞) ↔ (n : ℕ∞) ∈ s :=
  mem_nhds_iff (coe_ne_top _)


theorem tendsto_nhds_top_iff_natCast_lt {α : Type*} {l : Filter α} {f : α → ℕ∞} :
    Tendsto f l (𝓝 ⊤) ↔ ∀ n : ℕ, ∀ᶠ a in l, n < f a := by
  /-
    α : Type u_1
    l : Filter α
    f : α → ENat
    ⊢ Iff (Filter.Tendsto f l (nhds Top.top)) (∀ (n : Nat), Filter.Eventually (fun …
  -/
  simp_rw [nhds_top_order, lt_top_iff_ne_top, tendsto_iInf, tendsto_principal]
  /-
    α : Type u_1
    l : Filter α
    f : α → ENat
    ⊢ Iff (∀ (i : ENat), Ne i Top.top → Filter.Eventually (fun a => Membership.mem …
  -/
  exact Option.ball_ne_none
  /-
    🎉 no goals
  -/


instance : ContinuousAdd ℕ∞ := by
  /-
    ⊢ ContinuousAdd ENat
  -/
  refine ⟨continuous_iff_continuousAt.2 fun (a, b) ↦ ?_⟩
  match a, b with
  | ⊤, _ => exact tendsto_nhds_top_mono' continuousAt_fst fun p ↦ le_add_right le_rfl
  | (a : ℕ), ⊤ => exact tendsto_nhds_top_mono' continuousAt_snd fun p ↦ le_add_left le_rfl
  | (a : ℕ), (b : ℕ) => simp [ContinuousAt, nhds_prod_eq, tendsto_pure_nhds]


instance : ContinuousMul ℕ∞ where
  continuous_mul :=
    have key (a : ℕ∞) : ContinuousAt (· * ·).uncurry (a, ⊤) := by
      /-
        a : ENat
        ⊢ ContinuousAt (Function.uncurry fun x1 x2 => HMul.hMul x1 x2) { fst := a, snd …
      -/
      rcases (zero_le a).eq_or_gt with rfl | ha
        /-
          case inl
          ⊢ ContinuousAt (Function.uncurry fun x1 x2 => HMul.hMul x1 x2) { fst := 0, snd …
        -/
      · simp [ContinuousAt, nhds_prod_eq]
        /-
          🎉 no goals
        -/
        /-
          case inr
          a : ENat
          ha : LT.lt 0 a
          ⊢ ContinuousAt (Function.uncurry fun x1 x2 => HMul.hMul x1 x2) { fst := a, snd …
        -/
      · simp only [ContinuousAt, Function.uncurry, mul_top ha.ne']
        /-
          case inr
          a : ENat
          ha : LT.lt 0 a
          ⊢ Filter.Tendsto (Function.uncurry fun x1 x2 => HMul.hMul x1 x2) (nhds { fst : …
        -/
        refine tendsto_nhds_top_mono continuousAt_snd ?_
        /-
          case inr
          a : ENat
          ha : LT.lt 0 a
          ⊢ (nhds { fst := a, snd := Top.top }).EventuallyLE Prod.snd (Function.uncurry  …
        -/
        filter_upwards [continuousAt_fst (lt_mem_nhds ha)] with (x, y) (hx : 0 < x)
        /-
          case h
          a : ENat
          ha : LT.lt 0 a
          x y : ENat
          hx : LT.lt 0 x
          ⊢ LE.le { fst := x, snd := y }.2 (Function.uncurry (fun x1 x2 => HMul.hMul x1  …
        -/
        exact le_mul_of_one_le_left (zero_le y) (Order.one_le_iff_pos.2 hx)
        /-
          🎉 no goals
        -/
    continuous_iff_continuousAt.2 <| Prod.forall.2 fun
      | (a : ℕ∞), ⊤ => key a
      | ⊤, (b : ℕ∞) =>
        ((key b).comp_of_eq (continuous_swap.tendsto (⊤, b)) rfl).congr <|
          .of_forall fun _ ↦ mul_comm ..
      | (a : ℕ), (b : ℕ) => by
        /-
          key : ∀ (a : ENat), ContinuousAt (Function.uncurry fun x1 x2 => HMul.hMul x1 x …
          a b : Nat
          ⊢ ContinuousAt (fun p => HMul.hMul p.1 p.2) { fst := ↑a, snd := ↑b }
        -/
        simp [ContinuousAt, nhds_prod_eq, tendsto_pure_nhds]
        /-
          🎉 no goals
        -/


protected theorem continuousAt_sub {a b : ℕ∞} (h : a ≠ ⊤ ∨ b ≠ ⊤) :
    ContinuousAt (· - ·).uncurry (a, b) := by
  match a, b, h with
  | (a : ℕ), (b : ℕ), _ => simp [ContinuousAt, nhds_prod_eq]
  | (a : ℕ), ⊤, _ =>
    suffices ∀ᶠ b in 𝓝 ⊤, (a - b : ℕ∞) = 0 by
      simpa [ContinuousAt, nhds_prod_eq, tsub_eq_zero_of_le]
    filter_upwards [le_mem_nhds (WithTop.coe_lt_top a)] with b using tsub_eq_zero_of_le
  | ⊤, (b : ℕ), _ =>
    suffices ∀ n : ℕ, ∀ᶠ a : ℕ∞ in 𝓝 ⊤, b + n < a by
      simpa [ContinuousAt, nhds_prod_eq, (· ∘ ·), lt_tsub_iff_left, tendsto_nhds_top_iff_natCast_lt]
    exact fun n ↦ lt_mem_nhds <| WithTop.coe_lt_top (b + n)


theorem Filter.Tendsto.enatSub {α : Type*} {l : Filter α} {f g : α → ℕ∞} {a b : ℕ∞}
    (hf : Tendsto f l (𝓝 a)) (hg : Tendsto g l (𝓝 b)) (h : a ≠ ⊤ ∨ b ≠ ⊤) :
    Tendsto (fun x ↦ f x - g x) l (𝓝 (a - b)) :=
  (ENat.continuousAt_sub h).tendsto.comp (hf.prod_mk_nhds hg)


nonrec theorem ContinuousWithinAt.enatSub
    (hf : ContinuousWithinAt f s x) (hg : ContinuousWithinAt g s x) (h : f x ≠ ⊤ ∨ g x ≠ ⊤) :
    ContinuousWithinAt (fun x ↦ f x - g x) s x :=
  hf.enatSub hg h


nonrec theorem ContinuousAt.enatSub
    (hf : ContinuousAt f x) (hg : ContinuousAt g x) (h : f x ≠ ⊤ ∨ g x ≠ ⊤) :
    ContinuousAt (fun x ↦ f x - g x) x :=
  hf.enatSub hg h


nonrec theorem ContinuousOn.enatSub
    (hf : ContinuousOn f s) (hg : ContinuousOn g s) (h : ∀ x ∈ s, f x ≠ ⊤ ∨ g x ≠ ⊤) :
    ContinuousOn (fun x ↦ f x - g x) s := fun x hx ↦
  (hf x hx).enatSub (hg x hx) (h x hx)


nonrec theorem Continuous.enatSub
    (hf : Continuous f) (hg : Continuous g) (h : ∀ x, f x ≠ ⊤ ∨ g x ≠ ⊤) :
    Continuous (fun x ↦ f x - g x) :=
  continuous_iff_continuousAt.2 fun x ↦ hf.continuousAt.enatSub hg.continuousAt (h x)

