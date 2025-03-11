theorem tendsto_mul_pow_div_factorial_sub_atTop (a c : K) (d : ℕ) :
    Tendsto (fun n ↦ a * c ^ n / (n - d)!) atTop (𝓝 0) := by
  /-
    K : Type u_1
    inst✝³ : LinearOrderedField K
    inst✝² : FloorSemiring K
    inst✝¹ : TopologicalSpace K
    inst✝ : OrderTopology K
    a c : K
    d : Nat
    ⊢ Filter.Tendsto (fun n => HDiv.hDiv (HMul.hMul a (HPow.hPow c n)) ↑(HSub.hSub …
  -/
  rw [tendsto_order]
  /-
    K : Type u_1
    inst✝³ : LinearOrderedField K
    inst✝² : FloorSemiring K
    inst✝¹ : TopologicalSpace K
    inst✝ : OrderTopology K
    a c : K
    d : Nat
    ⊢ And (∀ (a' : K), LT.lt a' 0 → Filter.Eventually (fun b => LT.lt a' (HDiv.hDi …
  -/
  constructor
  all_goals
    intro ε hε
    filter_upwards [eventually_mul_pow_lt_factorial_sub (a * ε⁻¹) c d] with n h
    rw [mul_right_comm, ← div_eq_mul_inv] at h
    /-
      case h
      K : Type u_1
      inst✝³ : LinearOrderedField K
      inst✝² : FloorSemiring K
      inst✝¹ : TopologicalSpace K
      inst✝ : OrderTopology K
      a c : K
      d : Nat
      ε : K
      hε : LT.lt ε 0
      n : Nat
      h : LT.lt (HDiv.hDiv (HMul.hMul a (HPow.hPow c n)) ε) ↑(HSub.hSub n d).factorial
      ⊢ LT.lt ε (HDiv.hDiv (HMul.hMul a (HPow.hPow c n)) ↑(HSub.hSub n d).factorial)
    -/
  · rw [div_lt_iff_of_neg hε] at h
    /-
      case h
      K : Type u_1
      inst✝³ : LinearOrderedField K
      inst✝² : FloorSemiring K
      inst✝¹ : TopologicalSpace K
      inst✝ : OrderTopology K
      a c : K
      d : Nat
      ε : K
      hε : LT.lt ε 0
      n : Nat
      h : LT.lt (HMul.hMul (↑(HSub.hSub n d).factorial) ε) (HMul.hMul a (HPow.hPow c …
      ⊢ LT.lt ε (HDiv.hDiv (HMul.hMul a (HPow.hPow c n)) ↑(HSub.hSub n d).factorial)
    -/
    rwa [lt_div_iff₀' (Nat.cast_pos.mpr (Nat.factorial_pos _))]
    /-
      🎉 no goals
    -/
    /-
      case h
      K : Type u_1
      inst✝³ : LinearOrderedField K
      inst✝² : FloorSemiring K
      inst✝¹ : TopologicalSpace K
      inst✝ : OrderTopology K
      a c : K
      d : Nat
      ε : K
      hε : GT.gt ε 0
      n : Nat
      h : LT.lt (HDiv.hDiv (HMul.hMul a (HPow.hPow c n)) ε) ↑(HSub.hSub n d).factorial
      ⊢ LT.lt (HDiv.hDiv (HMul.hMul a (HPow.hPow c n)) ↑(HSub.hSub n d).factorial) ε
    -/
  · rw [div_lt_iff₀ hε] at h
    /-
      case h
      K : Type u_1
      inst✝³ : LinearOrderedField K
      inst✝² : FloorSemiring K
      inst✝¹ : TopologicalSpace K
      inst✝ : OrderTopology K
      a c : K
      d : Nat
      ε : K
      hε : GT.gt ε 0
      n : Nat
      h : LT.lt (HMul.hMul a (HPow.hPow c n)) (HMul.hMul (↑(HSub.hSub n d).factorial …
      ⊢ LT.lt (HDiv.hDiv (HMul.hMul a (HPow.hPow c n)) ↑(HSub.hSub n d).factorial) ε
    -/
    rwa [div_lt_iff₀' (Nat.cast_pos.mpr (Nat.factorial_pos _))]
    /-
      🎉 no goals
    -/


theorem tendsto_pow_div_factorial_atTop (c : K) :
    Tendsto (fun n ↦ c ^ n / n !) atTop (𝓝 0) := by
  /-
    K : Type u_1
    inst✝³ : LinearOrderedField K
    inst✝² : FloorSemiring K
    inst✝¹ : TopologicalSpace K
    inst✝ : OrderTopology K
    c : K
    ⊢ Filter.Tendsto (fun n => HDiv.hDiv (HPow.hPow c n) ↑n.factorial) Filter.atTo …
  -/
  convert tendsto_mul_pow_div_factorial_sub_atTop 1 c 0
  /-
    case h.e'_3.h.h.e'_5
    K : Type u_1
    inst✝³ : LinearOrderedField K
    inst✝² : FloorSemiring K
    inst✝¹ : TopologicalSpace K
    inst✝ : OrderTopology K
    c : K
    x✝ : Nat
    ⊢ Eq (HPow.hPow c x✝) (HMul.hMul 1 (HPow.hPow c x✝))
  -/
  rw [one_mul]
  /-
    🎉 no goals
  -/


theorem tendsto_floor_atTop : Tendsto (floor : α → ℤ) atTop atTop :=
  floor_mono.tendsto_atTop_atTop fun b =>
                     /-
                       α : Type u_1
                       inst✝¹ : LinearOrderedRing α
                       inst✝ : FloorRing α
                       b : Int
                       ⊢ LE.le b (Int.floor ↑(HAdd.hAdd b 1))
                     -/
    ⟨(b + 1 : ℤ), by rw [floor_intCast]; exact (lt_add_one _).le⟩
                                         /-
                                           🎉 no goals
                                         -/


theorem tendsto_floor_atBot : Tendsto (floor : α → ℤ) atBot atBot :=
  floor_mono.tendsto_atBot_atBot fun b => ⟨b, (floor_intCast _).le⟩


theorem tendsto_ceil_atTop : Tendsto (ceil : α → ℤ) atTop atTop :=
  ceil_mono.tendsto_atTop_atTop fun b => ⟨b, (ceil_intCast _).ge⟩


theorem tendsto_ceil_atBot : Tendsto (ceil : α → ℤ) atBot atBot :=
  ceil_mono.tendsto_atBot_atBot fun b =>
                     /-
                       α : Type u_1
                       inst✝¹ : LinearOrderedRing α
                       inst✝ : FloorRing α
                       b : Int
                       ⊢ LE.le (Int.ceil ↑(HSub.hSub b 1)) b
                     -/
    ⟨(b - 1 : ℤ), by rw [ceil_intCast]; exact (sub_one_lt _).le⟩
                                        /-
                                          🎉 no goals
                                        -/


theorem continuousOn_floor (n : ℤ) :
    ContinuousOn (fun x => floor x : α → α) (Ico n (n + 1) : Set α) :=
  (continuousOn_congr <| floor_eq_on_Ico' n).mpr continuousOn_const


theorem continuousOn_ceil (n : ℤ) :
    ContinuousOn (fun x => ceil x : α → α) (Ioc (n - 1) n : Set α) :=
  (continuousOn_congr <| ceil_eq_on_Ioc' n).mpr continuousOn_const


theorem tendsto_floor_right_pure_floor (x : α) : Tendsto (floor : α → ℤ) (𝓝[≥] x) (pure ⌊x⌋) :=
  tendsto_pure.2 <| mem_of_superset (Ico_mem_nhdsGE <| lt_floor_add_one x) fun _y hy =>
    floor_eq_on_Ico _ _ ⟨(floor_le x).trans hy.1, hy.2⟩


theorem tendsto_floor_right_pure (n : ℤ) : Tendsto (floor : α → ℤ) (𝓝[≥] n) (pure n) := by
  /-
    α : Type u_1
    inst✝³ : LinearOrderedRing α
    inst✝² : FloorRing α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderClosedTopology α
    n : Int
    ⊢ Filter.Tendsto Int.floor (nhdsWithin (↑n) (Set.Ici ↑n)) (Pure.pure n)
  -/
  simpa only [floor_intCast] using tendsto_floor_right_pure_floor (n : α)
  /-
    🎉 no goals
  -/


theorem tendsto_ceil_left_pure_ceil (x : α) : Tendsto (ceil : α → ℤ) (𝓝[≤] x) (pure ⌈x⌉) :=
  tendsto_pure.2 <| mem_of_superset
    (Ioc_mem_nhdsLE <| sub_lt_iff_lt_add.2 <| ceil_lt_add_one _) fun _y hy =>
      ceil_eq_on_Ioc _ _ ⟨hy.1, hy.2.trans (le_ceil _)⟩


theorem tendsto_ceil_left_pure (n : ℤ) : Tendsto (ceil : α → ℤ) (𝓝[≤] n) (pure n) := by
  /-
    α : Type u_1
    inst✝³ : LinearOrderedRing α
    inst✝² : FloorRing α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderClosedTopology α
    n : Int
    ⊢ Filter.Tendsto Int.ceil (nhdsWithin (↑n) (Set.Iic ↑n)) (Pure.pure n)
  -/
  simpa only [ceil_intCast] using tendsto_ceil_left_pure_ceil (n : α)
  /-
    🎉 no goals
  -/


theorem tendsto_floor_left_pure_ceil_sub_one (x : α) :
    Tendsto (floor : α → ℤ) (𝓝[<] x) (pure (⌈x⌉ - 1)) :=
                                 /-
                                   α : Type u_1
                                   inst✝³ : LinearOrderedRing α
                                   inst✝² : FloorRing α
                                   inst✝¹ : TopologicalSpace α
                                   inst✝ : OrderClosedTopology α
                                   x : α
                                   ⊢ LT.lt (↑(HSub.hSub (Int.ceil x) 1)) x
                                 -/
  have h₁ : ↑(⌈x⌉ - 1) < x := by rw [cast_sub, cast_one, sub_lt_iff_lt_add]; exact ceil_lt_add_one _
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
                                     /-
                                       α : Type u_1
                                       inst✝³ : LinearOrderedRing α
                                       inst✝² : FloorRing α
                                       inst✝¹ : TopologicalSpace α
                                       inst✝ : OrderClosedTopology α
                                       x : α
                                       h₁ : LT.lt (↑(HSub.hSub (Int.ceil x) 1)) x
                                       ⊢ LE.le x (HAdd.hAdd (↑(HSub.hSub (Int.ceil x) 1)) 1)
                                     -/
  have h₂ : x ≤ ↑(⌈x⌉ - 1) + 1 := by rw [cast_sub, cast_one, sub_add_cancel]; exact le_ceil _
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
  tendsto_pure.2 <| mem_of_superset (Ico_mem_nhdsLT h₁) fun _y hy =>
    floor_eq_on_Ico _ _ ⟨hy.1, hy.2.trans_le h₂⟩


theorem tendsto_floor_left_pure_sub_one (n : ℤ) :
    Tendsto (floor : α → ℤ) (𝓝[<] n) (pure (n - 1)) := by
  /-
    α : Type u_1
    inst✝³ : LinearOrderedRing α
    inst✝² : FloorRing α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderClosedTopology α
    n : Int
    ⊢ Filter.Tendsto Int.floor (nhdsWithin (↑n) (Set.Iio ↑n)) (Pure.pure (HSub.hSu …
  -/
  simpa only [ceil_intCast] using tendsto_floor_left_pure_ceil_sub_one (n : α)
  /-
    🎉 no goals
  -/


theorem tendsto_ceil_right_pure_floor_add_one (x : α) :
    Tendsto (ceil : α → ℤ) (𝓝[>] x) (pure (⌊x⌋ + 1)) :=
                                  /-
                                    α : Type u_1
                                    inst✝³ : LinearOrderedRing α
                                    inst✝² : FloorRing α
                                    inst✝¹ : TopologicalSpace α
                                    inst✝ : OrderClosedTopology α
                                    x : α
                                    ⊢ LE.le (HSub.hSub (↑(HAdd.hAdd (Int.floor x) 1)) 1) x
                                  -/
  have : ↑(⌊x⌋ + 1) - 1 ≤ x := by rw [cast_add, cast_one, add_sub_cancel_right]; exact floor_le _
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
  tendsto_pure.2 <| mem_of_superset (Ioc_mem_nhdsGT <| lt_succ_floor _) fun _y hy =>
    ceil_eq_on_Ioc _ _ ⟨this.trans_lt hy.1, hy.2⟩


theorem tendsto_ceil_right_pure_add_one (n : ℤ) :
    Tendsto (ceil : α → ℤ) (𝓝[>] n) (pure (n + 1)) := by
  /-
    α : Type u_1
    inst✝³ : LinearOrderedRing α
    inst✝² : FloorRing α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderClosedTopology α
    n : Int
    ⊢ Filter.Tendsto Int.ceil (nhdsWithin (↑n) (Set.Ioi ↑n)) (Pure.pure (HAdd.hAdd …
  -/
  simpa only [floor_intCast] using tendsto_ceil_right_pure_floor_add_one (n : α)
  /-
    🎉 no goals
  -/


theorem tendsto_floor_right (n : ℤ) : Tendsto (fun x => floor x : α → α) (𝓝[≥] n) (𝓝[≥] n) :=
  ((tendsto_pure_pure _ _).comp (tendsto_floor_right_pure n)).mono_right <|
    pure_le_nhdsWithin le_rfl


theorem tendsto_floor_right' (n : ℤ) : Tendsto (fun x => floor x : α → α) (𝓝[≥] n) (𝓝 n) :=
  (tendsto_floor_right n).mono_right inf_le_left


theorem tendsto_ceil_left (n : ℤ) : Tendsto (fun x => ceil x : α → α) (𝓝[≤] n) (𝓝[≤] n) :=
  ((tendsto_pure_pure _ _).comp (tendsto_ceil_left_pure n)).mono_right <|
    pure_le_nhdsWithin le_rfl


theorem tendsto_ceil_left' (n : ℤ) :
    Tendsto (fun x => ceil x : α → α) (𝓝[≤] n) (𝓝 n) :=
  (tendsto_ceil_left n).mono_right inf_le_left


theorem tendsto_floor_left (n : ℤ) :
    Tendsto (fun x => floor x : α → α) (𝓝[<] n) (𝓝[≤] (n - 1)) :=
  ((tendsto_pure_pure _ _).comp (tendsto_floor_left_pure_sub_one n)).mono_right <| by
    /-
      α : Type u_1
      inst✝³ : LinearOrderedRing α
      inst✝² : FloorRing α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderClosedTopology α
      n : Int
      ⊢ LE.le (Pure.pure (IntCast.intCast (HSub.hSub n 1))) (nhdsWithin (HSub.hSub ( …
    -/
    rw [← @cast_one α, ← cast_sub]; exact pure_le_nhdsWithin le_rfl
                                    /-
                                      🎉 no goals
                                    -/


theorem tendsto_ceil_right (n : ℤ) :
    Tendsto (fun x => ceil x : α → α) (𝓝[>] n) (𝓝[≥] (n + 1)) :=
  ((tendsto_pure_pure _ _).comp (tendsto_ceil_right_pure_add_one n)).mono_right <| by
    /-
      α : Type u_1
      inst✝³ : LinearOrderedRing α
      inst✝² : FloorRing α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderClosedTopology α
      n : Int
      ⊢ LE.le (Pure.pure (IntCast.intCast (HAdd.hAdd n 1))) (nhdsWithin (HAdd.hAdd ( …
    -/
    rw [← @cast_one α, ← cast_add]; exact pure_le_nhdsWithin le_rfl
                                    /-
                                      🎉 no goals
                                    -/


theorem tendsto_floor_left' (n : ℤ) :
    Tendsto (fun x => floor x : α → α) (𝓝[<] n) (𝓝 (n - 1)) :=
  (tendsto_floor_left n).mono_right inf_le_left


theorem tendsto_ceil_right' (n : ℤ) :
    Tendsto (fun x => ceil x : α → α) (𝓝[>] n) (𝓝 (n + 1)) :=
  (tendsto_ceil_right n).mono_right inf_le_left


theorem continuousOn_fract [TopologicalAddGroup α] (n : ℤ) :
    ContinuousOn (fract : α → α) (Ico n (n + 1) : Set α) :=
  continuousOn_id.sub (continuousOn_floor n)


theorem continuousAt_fract [OrderClosedTopology α] [TopologicalAddGroup α]
    {x : α} (h : x ≠ ⌊x⌋) : ContinuousAt fract x :=
  (continuousOn_fract ⌊x⌋).continuousAt <|
    Ico_mem_nhds ((floor_le _).lt_of_ne h.symm) (lt_floor_add_one _)


theorem tendsto_fract_left' [OrderClosedTopology α] [TopologicalAddGroup α] (n : ℤ) :
    Tendsto (fract : α → α) (𝓝[<] n) (𝓝 1) := by
  /-
    α : Type u_1
    inst✝⁴ : LinearOrderedRing α
    inst✝³ : FloorRing α
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderClosedTopology α
    inst✝ : TopologicalAddGroup α
    n : Int
    ⊢ Filter.Tendsto Int.fract (nhdsWithin (↑n) (Set.Iio ↑n)) (nhds 1)
  -/
  rw [← sub_sub_cancel (n : α) 1]
  /-
    α : Type u_1
    inst✝⁴ : LinearOrderedRing α
    inst✝³ : FloorRing α
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderClosedTopology α
    inst✝ : TopologicalAddGroup α
    n : Int
    ⊢ Filter.Tendsto Int.fract (nhdsWithin (↑n) (Set.Iio ↑n)) (nhds (HSub.hSub (↑n …
  -/
  refine (tendsto_id.mono_left nhdsWithin_le_nhds).sub ?_
  /-
    α : Type u_1
    inst✝⁴ : LinearOrderedRing α
    inst✝³ : FloorRing α
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderClosedTopology α
    inst✝ : TopologicalAddGroup α
    n : Int
    ⊢ Filter.Tendsto (fun x => ↑(Int.floor x)) (nhdsWithin (↑n) (Set.Iio ↑n)) (nhd …
  -/
  exact tendsto_floor_left' n
  /-
    🎉 no goals
  -/


theorem tendsto_fract_left [OrderClosedTopology α] [TopologicalAddGroup α] (n : ℤ) :
    Tendsto (fract : α → α) (𝓝[<] n) (𝓝[<] 1) :=
  tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within _ (tendsto_fract_left' _)
    (Eventually.of_forall fract_lt_one)


theorem tendsto_fract_right' [OrderClosedTopology α] [TopologicalAddGroup α] (n : ℤ) :
    Tendsto (fract : α → α) (𝓝[≥] n) (𝓝 0) :=
  sub_self (n : α) ▸ (tendsto_nhdsWithin_of_tendsto_nhds tendsto_id).sub (tendsto_floor_right' n)


theorem tendsto_fract_right [OrderClosedTopology α] [TopologicalAddGroup α] (n : ℤ) :
    Tendsto (fract : α → α) (𝓝[≥] n) (𝓝[≥] 0) :=
  tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within _ (tendsto_fract_right' _)
    (Eventually.of_forall fract_nonneg)


local notation "I" => (Icc 0 1 : Set α)


/-- Do not use this, use `ContinuousOn.comp_fract` instead. -/
theorem ContinuousOn.comp_fract' {f : β → α → γ} (h : ContinuousOn (uncurry f) <| univ ×ˢ I)
    (hf : ∀ s, f s 0 = f s 1) : Continuous fun st : β × α => f st.1 (fract st.2) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁵ : LinearOrderedRing α
    inst✝⁴ : FloorRing α
    inst✝³ : TopologicalSpace α
    inst✝² : OrderTopology α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    f : β → α → γ
    h : ContinuousOn (Function.uncurry f) (SProd.sprod Set.univ (Set.Icc 0 1))
    hf : ∀ (s : β), Eq (f s 0) (f s 1)
    ⊢ Continuous fun st => f st.1 (Int.fract st.2)
  -/
  change Continuous (uncurry f ∘ Prod.map id fract)
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁵ : LinearOrderedRing α
    inst✝⁴ : FloorRing α
    inst✝³ : TopologicalSpace α
    inst✝² : OrderTopology α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    f : β → α → γ
    h : ContinuousOn (Function.uncurry f) (SProd.sprod Set.univ (Set.Icc 0 1))
    hf : ∀ (s : β), Eq (f s 0) (f s 1)
    ⊢ Continuous (Function.comp (Function.uncurry f) (Prod.map id Int.fract))
  -/
  rw [continuous_iff_continuousAt]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁵ : LinearOrderedRing α
    inst✝⁴ : FloorRing α
    inst✝³ : TopologicalSpace α
    inst✝² : OrderTopology α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    f : β → α → γ
    h : ContinuousOn (Function.uncurry f) (SProd.sprod Set.univ (Set.Icc 0 1))
    hf : ∀ (s : β), Eq (f s 0) (f s 1)
    ⊢ ∀ (x : Prod β α), ContinuousAt (Function.comp (Function.uncurry f) (Prod.map …
  -/
  rintro ⟨s, t⟩
  /-
    case mk
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁵ : LinearOrderedRing α
    inst✝⁴ : FloorRing α
    inst✝³ : TopologicalSpace α
    inst✝² : OrderTopology α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    f : β → α → γ
    h : ContinuousOn (Function.uncurry f) (SProd.sprod Set.univ (Set.Icc 0 1))
    hf : ∀ (s : β), Eq (f s 0) (f s 1)
    s : β
    t : α
    ⊢ ContinuousAt (Function.comp (Function.uncurry f) (Prod.map id Int.fract)) {  …
  -/
  rcases em (∃ n : ℤ, t = n) with (⟨n, rfl⟩ | ht)
    /-
      case mk.inl.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁵ : LinearOrderedRing α
      inst✝⁴ : FloorRing α
      inst✝³ : TopologicalSpace α
      inst✝² : OrderTopology α
      inst✝¹ : TopologicalSpace β
      inst✝ : TopologicalSpace γ
      f : β → α → γ
      h : ContinuousOn (Function.uncurry f) (SProd.sprod Set.univ (Set.Icc 0 1))
      hf : ∀ (s : β), Eq (f s 0) (f s 1)
      s : β
      n : Int
      ⊢ ContinuousAt (Function.comp (Function.uncurry f) (Prod.map id Int.fract)) {  …
    -/
  · rw [ContinuousAt, nhds_prod_eq, ← nhdsLT_sup_nhdsGE (n : α), prod_sup, tendsto_sup]
    /-
      case mk.inl.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁵ : LinearOrderedRing α
      inst✝⁴ : FloorRing α
      inst✝³ : TopologicalSpace α
      inst✝² : OrderTopology α
      inst✝¹ : TopologicalSpace β
      inst✝ : TopologicalSpace γ
      f : β → α → γ
      h : ContinuousOn (Function.uncurry f) (SProd.sprod Set.univ (Set.Icc 0 1))
      hf : ∀ (s : β), Eq (f s 0) (f s 1)
      s : β
      n : Int
      ⊢ And (Filter.Tendsto (Function.comp (Function.uncurry f) (Prod.map id Int.fra …
    -/
    constructor
    · refine (((h (s, 1) ⟨trivial, zero_le_one, le_rfl⟩).tendsto.mono_left ?_).comp
        (tendsto_id.prod_map (tendsto_fract_left _))).mono_right (le_of_eq ?_)
        /-
          case mk.inl.intro.left.refine_1
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          inst✝⁵ : LinearOrderedRing α
          inst✝⁴ : FloorRing α
          inst✝³ : TopologicalSpace α
          inst✝² : OrderTopology α
          inst✝¹ : TopologicalSpace β
          inst✝ : TopologicalSpace γ
          f : β → α → γ
          h : ContinuousOn (Function.uncurry f) (SProd.sprod Set.univ (Set.Icc 0 1))
          hf : ∀ (s : β), Eq (f s 0) (f s 1)
          s : β
          n : Int
          ⊢ LE.le (SProd.sprod (nhds s) (nhdsWithin 1 (Set.Iio 1))) (nhdsWithin { fst := …
        -/
      · rw [nhdsWithin_prod_eq, nhdsWithin_univ, ← nhdsWithin_Ico_eq_nhdsLT one_pos]
        /-
          case mk.inl.intro.left.refine_1
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          inst✝⁵ : LinearOrderedRing α
          inst✝⁴ : FloorRing α
          inst✝³ : TopologicalSpace α
          inst✝² : OrderTopology α
          inst✝¹ : TopologicalSpace β
          inst✝ : TopologicalSpace γ
          f : β → α → γ
          h : ContinuousOn (Function.uncurry f) (SProd.sprod Set.univ (Set.Icc 0 1))
          hf : ∀ (s : β), Eq (f s 0) (f s 1)
          s : β
          n : Int
          ⊢ LE.le (SProd.sprod (nhds s) (nhdsWithin 1 (Set.Ico 0 1))) (SProd.sprod (nhds …
        -/
        exact Filter.prod_mono le_rfl (nhdsWithin_mono _ Ico_subset_Icc_self)
        /-
          🎉 no goals
        -/
        /-
          case mk.inl.intro.left.refine_2
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          inst✝⁵ : LinearOrderedRing α
          inst✝⁴ : FloorRing α
          inst✝³ : TopologicalSpace α
          inst✝² : OrderTopology α
          inst✝¹ : TopologicalSpace β
          inst✝ : TopologicalSpace γ
          f : β → α → γ
          h : ContinuousOn (Function.uncurry f) (SProd.sprod Set.univ (Set.Icc 0 1))
          hf : ∀ (s : β), Eq (f s 0) (f s 1)
          s : β
          n : Int
          ⊢ Eq (nhds (Function.uncurry f { fst := s, snd := 1 })) (nhds (Function.comp ( …
        -/
      · simp [hf]
        /-
          🎉 no goals
        -/
    · refine (((h (s, 0) ⟨trivial, le_rfl, zero_le_one⟩).tendsto.mono_left <| le_of_eq ?_).comp
        (tendsto_id.prod_map (tendsto_fract_right _))).mono_right (le_of_eq ?_) <;>
        /-
          case mk.inl.intro.right.refine_1
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          inst✝⁵ : LinearOrderedRing α
          inst✝⁴ : FloorRing α
          inst✝³ : TopologicalSpace α
          inst✝² : OrderTopology α
          inst✝¹ : TopologicalSpace β
          inst✝ : TopologicalSpace γ
          f : β → α → γ
          h : ContinuousOn (Function.uncurry f) (SProd.sprod Set.univ (Set.Icc 0 1))
          hf : ∀ (s : β), Eq (f s 0) (f s 1)
          s : β
          n : Int
          ⊢ Eq (SProd.sprod (nhds s) (nhdsWithin 0 (Set.Ici 0))) (nhdsWithin { fst := s, …
        -/
        /-
          🎉 no goals
        -/
        simp [nhdsWithin_prod_eq, nhdsWithin_univ]
        /-
          🎉 no goals
        -/
    /-
      case mk.inr
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁵ : LinearOrderedRing α
      inst✝⁴ : FloorRing α
      inst✝³ : TopologicalSpace α
      inst✝² : OrderTopology α
      inst✝¹ : TopologicalSpace β
      inst✝ : TopologicalSpace γ
      f : β → α → γ
      h : ContinuousOn (Function.uncurry f) (SProd.sprod Set.univ (Set.Icc 0 1))
      hf : ∀ (s : β), Eq (f s 0) (f s 1)
      s : β
      t : α
      ht : Not (Exists fun n => Eq t ↑n)
      ⊢ ContinuousAt (Function.comp (Function.uncurry f) (Prod.map id Int.fract)) {  …
    -/
  · replace ht : t ≠ ⌊t⌋ := fun ht' => ht ⟨_, ht'⟩
    /-
      case mk.inr
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁵ : LinearOrderedRing α
      inst✝⁴ : FloorRing α
      inst✝³ : TopologicalSpace α
      inst✝² : OrderTopology α
      inst✝¹ : TopologicalSpace β
      inst✝ : TopologicalSpace γ
      f : β → α → γ
      h : ContinuousOn (Function.uncurry f) (SProd.sprod Set.univ (Set.Icc 0 1))
      hf : ∀ (s : β), Eq (f s 0) (f s 1)
      s : β
      t : α
      ht : Ne t ↑(Int.floor t)
      ⊢ ContinuousAt (Function.comp (Function.uncurry f) (Prod.map id Int.fract)) {  …
    -/
    refine (h.continuousAt ?_).comp (continuousAt_id.prodMap (continuousAt_fract ht))
    /-
      case mk.inr
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁵ : LinearOrderedRing α
      inst✝⁴ : FloorRing α
      inst✝³ : TopologicalSpace α
      inst✝² : OrderTopology α
      inst✝¹ : TopologicalSpace β
      inst✝ : TopologicalSpace γ
      f : β → α → γ
      h : ContinuousOn (Function.uncurry f) (SProd.sprod Set.univ (Set.Icc 0 1))
      hf : ∀ (s : β), Eq (f s 0) (f s 1)
      s : β
      t : α
      ht : Ne t ↑(Int.floor t)
      ⊢ Membership.mem (nhds (Prod.map id Int.fract { fst := s, snd := t })) (SProd. …
    -/
    exact prod_mem_nhds univ_mem (Icc_mem_nhds (fract_pos.2 ht) (fract_lt_one _))
    /-
      🎉 no goals
    -/


theorem ContinuousOn.comp_fract {s : β → α} {f : β → α → γ}
    (h : ContinuousOn (uncurry f) <| univ ×ˢ Icc 0 1) (hs : Continuous s)
    (hf : ∀ s, f s 0 = f s 1) : Continuous fun x : β => f x <| Int.fract (s x) :=
  (h.comp_fract' hf).comp (continuous_id.prod_mk hs)


/-- A special case of `ContinuousOn.comp_fract`. -/
theorem ContinuousOn.comp_fract'' {f : α → β} (h : ContinuousOn f I) (hf : f 0 = f 1) :
    Continuous (f ∘ fract) :=
  ContinuousOn.comp_fract (h.comp continuousOn_snd fun _x hx => (mem_prod.mp hx).2) continuous_id
    fun _ => hf

