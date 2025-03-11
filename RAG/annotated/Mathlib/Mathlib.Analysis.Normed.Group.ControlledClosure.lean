/-- Given `f : NormedAddGroupHom G H` for some complete `G` and a subgroup `K` of `H`, if every
element `x` of `K` has a preimage under `f` whose norm is at most `C*‖x‖` then the same holds for
elements of the (topological) closure of `K` with constant `C+ε` instead of `C`, for any
positive `ε`.
-/
theorem controlled_closure_of_complete {f : NormedAddGroupHom G H} {K : AddSubgroup H} {C ε : ℝ}
    (hC : 0 < C) (hε : 0 < ε) (hyp : f.SurjectiveOnWith K C) :
    f.SurjectiveOnWith K.topologicalClosure (C + ε) := by
  /-
    G : Type u_1
    inst✝² : NormedAddCommGroup G
    inst✝¹ : CompleteSpace G
    H : Type u_2
    inst✝ : NormedAddCommGroup H
    f : NormedAddGroupHom G H
    K : AddSubgroup H
    C ε : Real
    hC : LT.lt 0 C
    hε : LT.lt 0 ε
    hyp : f.SurjectiveOnWith K C
    ⊢ f.SurjectiveOnWith K.topologicalClosure (HAdd.hAdd C ε)
  -/
  rintro (h : H) (h_in : h ∈ K.topologicalClosure)
  -- We first get rid of the easy case where `h = 0`.
  /-
    G : Type u_1
    inst✝² : NormedAddCommGroup G
    inst✝¹ : CompleteSpace G
    H : Type u_2
    inst✝ : NormedAddCommGroup H
    f : NormedAddGroupHom G H
    K : AddSubgroup H
    C ε : Real
    hC : LT.lt 0 C
    hε : LT.lt 0 ε
    hyp : f.SurjectiveOnWith K C
    h : H
    h_in : Membership.mem K.topologicalClosure h
    ⊢ Exists fun g => And (Eq (f g) h) (LE.le (Norm.norm g) (HMul.hMul (HAdd.hAdd  …
  -/
  by_cases hyp_h : h = 0
    /-
      case pos
      G : Type u_1
      inst✝² : NormedAddCommGroup G
      inst✝¹ : CompleteSpace G
      H : Type u_2
      inst✝ : NormedAddCommGroup H
      f : NormedAddGroupHom G H
      K : AddSubgroup H
      C ε : Real
      hC : LT.lt 0 C
      hε : LT.lt 0 ε
      hyp : f.SurjectiveOnWith K C
      h : H
      h_in : Membership.mem K.topologicalClosure h
      hyp_h : Eq h 0
      ⊢ Exists fun g => And (Eq (f g) h) (LE.le (Norm.norm g) (HMul.hMul (HAdd.hAdd  …
    -/
  · rw [hyp_h]
    /-
      case pos
      G : Type u_1
      inst✝² : NormedAddCommGroup G
      inst✝¹ : CompleteSpace G
      H : Type u_2
      inst✝ : NormedAddCommGroup H
      f : NormedAddGroupHom G H
      K : AddSubgroup H
      C ε : Real
      hC : LT.lt 0 C
      hε : LT.lt 0 ε
      hyp : f.SurjectiveOnWith K C
      h : H
      h_in : Membership.mem K.topologicalClosure h
      hyp_h : Eq h 0
      ⊢ Exists fun g => And (Eq (f g) 0) (LE.le (Norm.norm g) (HMul.hMul (HAdd.hAdd  …
    -/
    use 0
    /-
      case h
      G : Type u_1
      inst✝² : NormedAddCommGroup G
      inst✝¹ : CompleteSpace G
      H : Type u_2
      inst✝ : NormedAddCommGroup H
      f : NormedAddGroupHom G H
      K : AddSubgroup H
      C ε : Real
      hC : LT.lt 0 C
      hε : LT.lt 0 ε
      hyp : f.SurjectiveOnWith K C
      h : H
      h_in : Membership.mem K.topologicalClosure h
      hyp_h : Eq h 0
      ⊢ And (Eq (f 0) 0) (LE.le (Norm.norm 0) (HMul.hMul (HAdd.hAdd C ε) (Norm.norm  …
    -/
    simp
    /-
      🎉 no goals
    -/
  /- The desired preimage will be constructed as the sum of a series. Convergence of
    the series will be guaranteed by completeness of `G`. We first write `h` as the sum
    of a sequence `v` of elements of `K` which starts close to `h` and then quickly goes to zero.
    The sequence `b` below quantifies this. -/
  /-
    case neg
    G : Type u_1
    inst✝² : NormedAddCommGroup G
    inst✝¹ : CompleteSpace G
    H : Type u_2
    inst✝ : NormedAddCommGroup H
    f : NormedAddGroupHom G H
    K : AddSubgroup H
    C ε : Real
    hC : LT.lt 0 C
    hε : LT.lt 0 ε
    hyp : f.SurjectiveOnWith K C
    h : H
    h_in : Membership.mem K.topologicalClosure h
    hyp_h : Not (Eq h 0)
    ⊢ Exists fun g => And (Eq (f g) h) (LE.le (Norm.norm g) (HMul.hMul (HAdd.hAdd  …
  -/
  set b : ℕ → ℝ := fun i => (1 / 2) ^ i * (ε * ‖h‖ / 2) / C
  /-
    case neg
    G : Type u_1
    inst✝² : NormedAddCommGroup G
    inst✝¹ : CompleteSpace G
    H : Type u_2
    inst✝ : NormedAddCommGroup H
    f : NormedAddGroupHom G H
    K : AddSubgroup H
    C ε : Real
    hC : LT.lt 0 C
    hε : LT.lt 0 ε
    hyp : f.SurjectiveOnWith K C
    h : H
    h_in : Membership.mem K.topologicalClosure h
    hyp_h : Not (Eq h 0)
    b : Nat → Real := fun i => HDiv.hDiv (HMul.hMul (HPow.hPow (1 / 2) i) (HDiv.hD …
    ⊢ Exists fun g => And (Eq (f g) h) (LE.le (Norm.norm g) (HMul.hMul (HAdd.hAdd  …
  -/
  have b_pos (i) : 0 < b i := by field_simp [b, hC, hyp_h]
  obtain
    ⟨v : ℕ → H, lim_v : Tendsto (fun n : ℕ => ∑ k ∈ range (n + 1), v k) atTop (𝓝 h), v_in :
      ∀ n, v n ∈ K, hv₀ : ‖v 0 - h‖ < b 0, hv : ∀ n > 0, ‖v n‖ < b n⟩ :=
    controlled_sum_of_mem_closure h_in b_pos
  /- The controlled surjectivity assumption on `f` allows to build preimages `u n` for all
    elements `v n` of the `v` sequence. -/
  /-
    case neg.intro.intro.intro.intro
    G : Type u_1
    inst✝² : NormedAddCommGroup G
    inst✝¹ : CompleteSpace G
    H : Type u_2
    inst✝ : NormedAddCommGroup H
    f : NormedAddGroupHom G H
    K : AddSubgroup H
    C ε : Real
    hC : LT.lt 0 C
    hε : LT.lt 0 ε
    hyp : f.SurjectiveOnWith K C
    h : H
    h_in : Membership.mem K.topologicalClosure h
    hyp_h : Not (Eq h 0)
    b : Nat → Real := fun i => HDiv.hDiv (HMul.hMul (HPow.hPow (1 / 2) i) (HDiv.hD …
    b_pos : ∀ (i : Nat), LT.lt 0 (b i)
    v : Nat → H
    lim_v : Filter.Tendsto (fun n => (Finset.range (HAdd.hAdd n 1)).sum fun k => v …
    v_in : ∀ (n : Nat), Membership.mem K (v n)
    hv₀ : LT.lt (Norm.norm (HSub.hSub (v 0) h)) (b 0)
    hv : ∀ (n : Nat), GT.gt n 0 → LT.lt (Norm.norm (v n)) (b n)
    ⊢ Exists fun g => And (Eq (f g) h) (LE.le (Norm.norm g) (HMul.hMul (HAdd.hAdd  …
  -/
  have : ∀ n, ∃ m' : G, f m' = v n ∧ ‖m'‖ ≤ C * ‖v n‖ := fun n : ℕ => hyp (v n) (v_in n)
  /-
    case neg.intro.intro.intro.intro
    G : Type u_1
    inst✝² : NormedAddCommGroup G
    inst✝¹ : CompleteSpace G
    H : Type u_2
    inst✝ : NormedAddCommGroup H
    f : NormedAddGroupHom G H
    K : AddSubgroup H
    C ε : Real
    hC : LT.lt 0 C
    hε : LT.lt 0 ε
    hyp : f.SurjectiveOnWith K C
    h : H
    h_in : Membership.mem K.topologicalClosure h
    hyp_h : Not (Eq h 0)
    b : Nat → Real := fun i => HDiv.hDiv (HMul.hMul (HPow.hPow (1 / 2) i) (HDiv.hD …
    b_pos : ∀ (i : Nat), LT.lt 0 (b i)
    v : Nat → H
    lim_v : Filter.Tendsto (fun n => (Finset.range (HAdd.hAdd n 1)).sum fun k => v …
    v_in : ∀ (n : Nat), Membership.mem K (v n)
    hv₀ : LT.lt (Norm.norm (HSub.hSub (v 0) h)) (b 0)
    hv : ∀ (n : Nat), GT.gt n 0 → LT.lt (Norm.norm (v n)) (b n)
    this : ∀ (n : Nat), Exists fun m' => And (Eq (f m') (v n)) (LE.le (Norm.norm m …
    ⊢ Exists fun g => And (Eq (f g) h) (LE.le (Norm.norm g) (HMul.hMul (HAdd.hAdd  …
  -/
  choose u hu hnorm_u using this
  /- The desired series `s` is then obtained by summing `u`. We then check our choice of
    `b` ensures `s` is Cauchy. -/
  /-
    case neg.intro.intro.intro.intro
    G : Type u_1
    inst✝² : NormedAddCommGroup G
    inst✝¹ : CompleteSpace G
    H : Type u_2
    inst✝ : NormedAddCommGroup H
    f : NormedAddGroupHom G H
    K : AddSubgroup H
    C ε : Real
    hC : LT.lt 0 C
    hε : LT.lt 0 ε
    hyp : f.SurjectiveOnWith K C
    h : H
    h_in : Membership.mem K.topologicalClosure h
    hyp_h : Not (Eq h 0)
    b : Nat → Real := fun i => HDiv.hDiv (HMul.hMul (HPow.hPow (1 / 2) i) (HDiv.hD …
    b_pos : ∀ (i : Nat), LT.lt 0 (b i)
    v : Nat → H
    lim_v : Filter.Tendsto (fun n => (Finset.range (HAdd.hAdd n 1)).sum fun k => v …
    v_in : ∀ (n : Nat), Membership.mem K (v n)
    hv₀ : LT.lt (Norm.norm (HSub.hSub (v 0) h)) (b 0)
    hv : ∀ (n : Nat), GT.gt n 0 → LT.lt (Norm.norm (v n)) (b n)
    u : Nat → G
    hu : ∀ (n : Nat), Eq (f (u n)) (v n)
    hnorm_u : ∀ (n : Nat), LE.le (Norm.norm (u n)) (HMul.hMul C (Norm.norm (v n)))
    ⊢ Exists fun g => And (Eq (f g) h) (LE.le (Norm.norm g) (HMul.hMul (HAdd.hAdd  …
  -/
  set s : ℕ → G := fun n => ∑ k ∈ range (n + 1), u k
  have : CauchySeq s := by
    apply NormedAddCommGroup.cauchy_series_of_le_geometric'' (by norm_num) one_half_lt_one
    · rintro n (hn : n ≥ 1)
      calc
        ‖u n‖ ≤ C * ‖v n‖ := hnorm_u n
        _ ≤ C * b n := by gcongr; exact (hv _ <| Nat.succ_le_iff.mp hn).le
        _ = (1 / 2) ^ n * (ε * ‖h‖ / 2) := by simp [b, mul_div_cancel₀ _ hC.ne.symm]
        _ = ε * ‖h‖ / 2 * (1 / 2) ^ n := mul_comm _ _
  -- We now show that the limit `g` of `s` is the desired preimage.
  /-
    case neg.intro.intro.intro.intro
    G : Type u_1
    inst✝² : NormedAddCommGroup G
    inst✝¹ : CompleteSpace G
    H : Type u_2
    inst✝ : NormedAddCommGroup H
    f : NormedAddGroupHom G H
    K : AddSubgroup H
    C ε : Real
    hC : LT.lt 0 C
    hε : LT.lt 0 ε
    hyp : f.SurjectiveOnWith K C
    h : H
    h_in : Membership.mem K.topologicalClosure h
    hyp_h : Not (Eq h 0)
    b : Nat → Real := fun i => HDiv.hDiv (HMul.hMul (HPow.hPow (1 / 2) i) (HDiv.hD …
    b_pos : ∀ (i : Nat), LT.lt 0 (b i)
    v : Nat → H
    lim_v : Filter.Tendsto (fun n => (Finset.range (HAdd.hAdd n 1)).sum fun k => v …
    v_in : ∀ (n : Nat), Membership.mem K (v n)
    hv₀ : LT.lt (Norm.norm (HSub.hSub (v 0) h)) (b 0)
    hv : ∀ (n : Nat), GT.gt n 0 → LT.lt (Norm.norm (v n)) (b n)
    u : Nat → G
    hu : ∀ (n : Nat), Eq (f (u n)) (v n)
    hnorm_u : ∀ (n : Nat), LE.le (Norm.norm (u n)) (HMul.hMul C (Norm.norm (v n)))
    s : Nat → G := fun n => (Finset.range (HAdd.hAdd n 1)).sum fun k => u k
    this : CauchySeq s
    ⊢ Exists fun g => And (Eq (f g) h) (LE.le (Norm.norm g) (HMul.hMul (HAdd.hAdd  …
  -/
  obtain ⟨g : G, hg⟩ := cauchySeq_tendsto_of_complete this
  /-
    case neg.intro.intro.intro.intro.intro
    G : Type u_1
    inst✝² : NormedAddCommGroup G
    inst✝¹ : CompleteSpace G
    H : Type u_2
    inst✝ : NormedAddCommGroup H
    f : NormedAddGroupHom G H
    K : AddSubgroup H
    C ε : Real
    hC : LT.lt 0 C
    hε : LT.lt 0 ε
    hyp : f.SurjectiveOnWith K C
    h : H
    h_in : Membership.mem K.topologicalClosure h
    hyp_h : Not (Eq h 0)
    b : Nat → Real := fun i => HDiv.hDiv (HMul.hMul (HPow.hPow (1 / 2) i) (HDiv.hD …
    b_pos : ∀ (i : Nat), LT.lt 0 (b i)
    v : Nat → H
    lim_v : Filter.Tendsto (fun n => (Finset.range (HAdd.hAdd n 1)).sum fun k => v …
    v_in : ∀ (n : Nat), Membership.mem K (v n)
    hv₀ : LT.lt (Norm.norm (HSub.hSub (v 0) h)) (b 0)
    hv : ∀ (n : Nat), GT.gt n 0 → LT.lt (Norm.norm (v n)) (b n)
    u : Nat → G
    hu : ∀ (n : Nat), Eq (f (u n)) (v n)
    hnorm_u : ∀ (n : Nat), LE.le (Norm.norm (u n)) (HMul.hMul C (Norm.norm (v n)))
    s : Nat → G := fun n => (Finset.range (HAdd.hAdd n 1)).sum fun k => u k
    this : CauchySeq s
    g : G
    hg : Filter.Tendsto s Filter.atTop (nhds g)
    ⊢ Exists fun g => And (Eq (f g) h) (LE.le (Norm.norm g) (HMul.hMul (HAdd.hAdd  …
  -/
  refine ⟨g, ?_, ?_⟩
  · -- We indeed get a preimage. First note:
    have : f ∘ s = fun n => ∑ k ∈ range (n + 1), v k := by
      ext n
      simp [s, map_sum, hu]
    /- In the above equality, the left-hand-side converges to `f g` by continuity of `f` and
      definition of `g` while the right-hand-side converges to `h` by construction of `v` so
      `g` is indeed a preimage of `h`. -/
    /-
      case neg.intro.intro.intro.intro.intro.refine_1
      G : Type u_1
      inst✝² : NormedAddCommGroup G
      inst✝¹ : CompleteSpace G
      H : Type u_2
      inst✝ : NormedAddCommGroup H
      f : NormedAddGroupHom G H
      K : AddSubgroup H
      C ε : Real
      hC : LT.lt 0 C
      hε : LT.lt 0 ε
      hyp : f.SurjectiveOnWith K C
      h : H
      h_in : Membership.mem K.topologicalClosure h
      hyp_h : Not (Eq h 0)
      b : Nat → Real := fun i => HDiv.hDiv (HMul.hMul (HPow.hPow (1 / 2) i) (HDiv.hD …
      b_pos : ∀ (i : Nat), LT.lt 0 (b i)
      v : Nat → H
      lim_v : Filter.Tendsto (fun n => (Finset.range (HAdd.hAdd n 1)).sum fun k => v …
      v_in : ∀ (n : Nat), Membership.mem K (v n)
      hv₀ : LT.lt (Norm.norm (HSub.hSub (v 0) h)) (b 0)
      hv : ∀ (n : Nat), GT.gt n 0 → LT.lt (Norm.norm (v n)) (b n)
      u : Nat → G
      hu : ∀ (n : Nat), Eq (f (u n)) (v n)
      hnorm_u : ∀ (n : Nat), LE.le (Norm.norm (u n)) (HMul.hMul C (Norm.norm (v n)))
      s : Nat → G := fun n => (Finset.range (HAdd.hAdd n 1)).sum fun k => u k
      this✝ : CauchySeq s
      g : G
      hg : Filter.Tendsto s Filter.atTop (nhds g)
      this : Eq (Function.comp (⇑f) s) fun n => (Finset.range (HAdd.hAdd n 1)).sum f …
      ⊢ Eq (f g) h
    -/
    rw [← this] at lim_v
    /-
      case neg.intro.intro.intro.intro.intro.refine_1
      G : Type u_1
      inst✝² : NormedAddCommGroup G
      inst✝¹ : CompleteSpace G
      H : Type u_2
      inst✝ : NormedAddCommGroup H
      f : NormedAddGroupHom G H
      K : AddSubgroup H
      C ε : Real
      hC : LT.lt 0 C
      hε : LT.lt 0 ε
      hyp : f.SurjectiveOnWith K C
      h : H
      h_in : Membership.mem K.topologicalClosure h
      hyp_h : Not (Eq h 0)
      b : Nat → Real := fun i => HDiv.hDiv (HMul.hMul (HPow.hPow (1 / 2) i) (HDiv.hD …
      b_pos : ∀ (i : Nat), LT.lt 0 (b i)
      v : Nat → H
      v_in : ∀ (n : Nat), Membership.mem K (v n)
      hv₀ : LT.lt (Norm.norm (HSub.hSub (v 0) h)) (b 0)
      hv : ∀ (n : Nat), GT.gt n 0 → LT.lt (Norm.norm (v n)) (b n)
      u : Nat → G
      hu : ∀ (n : Nat), Eq (f (u n)) (v n)
      hnorm_u : ∀ (n : Nat), LE.le (Norm.norm (u n)) (HMul.hMul C (Norm.norm (v n)))
      s : Nat → G := fun n => (Finset.range (HAdd.hAdd n 1)).sum fun k => u k
      lim_v : Filter.Tendsto (Function.comp (⇑f) s) Filter.atTop (nhds h)
      this✝ : CauchySeq s
      g : G
      hg : Filter.Tendsto s Filter.atTop (nhds g)
      this : Eq (Function.comp (⇑f) s) fun n => (Finset.range (HAdd.hAdd n 1)).sum f …
      ⊢ Eq (f g) h
    -/
    exact tendsto_nhds_unique ((f.continuous.tendsto g).comp hg) lim_v
    /-
      🎉 no goals
    -/
  · -- Then we need to estimate the norm of `g`, using our careful choice of `b`.
    suffices ∀ n, ‖s n‖ ≤ (C + ε) * ‖h‖ from
      le_of_tendsto' (continuous_norm.continuousAt.tendsto.comp hg) this
    /-
      case neg.intro.intro.intro.intro.intro.refine_2
      G : Type u_1
      inst✝² : NormedAddCommGroup G
      inst✝¹ : CompleteSpace G
      H : Type u_2
      inst✝ : NormedAddCommGroup H
      f : NormedAddGroupHom G H
      K : AddSubgroup H
      C ε : Real
      hC : LT.lt 0 C
      hε : LT.lt 0 ε
      hyp : f.SurjectiveOnWith K C
      h : H
      h_in : Membership.mem K.topologicalClosure h
      hyp_h : Not (Eq h 0)
      b : Nat → Real := fun i => HDiv.hDiv (HMul.hMul (HPow.hPow (1 / 2) i) (HDiv.hD …
      b_pos : ∀ (i : Nat), LT.lt 0 (b i)
      v : Nat → H
      lim_v : Filter.Tendsto (fun n => (Finset.range (HAdd.hAdd n 1)).sum fun k => v …
      v_in : ∀ (n : Nat), Membership.mem K (v n)
      hv₀ : LT.lt (Norm.norm (HSub.hSub (v 0) h)) (b 0)
      hv : ∀ (n : Nat), GT.gt n 0 → LT.lt (Norm.norm (v n)) (b n)
      u : Nat → G
      hu : ∀ (n : Nat), Eq (f (u n)) (v n)
      hnorm_u : ∀ (n : Nat), LE.le (Norm.norm (u n)) (HMul.hMul C (Norm.norm (v n)))
      s : Nat → G := fun n => (Finset.range (HAdd.hAdd n 1)).sum fun k => u k
      this : CauchySeq s
      g : G
      hg : Filter.Tendsto s Filter.atTop (nhds g)
      ⊢ ∀ (n : Nat), LE.le (Norm.norm (s n)) (HMul.hMul (HAdd.hAdd C ε) (Norm.norm h))
    -/
    intro n
    have hnorm₀ : ‖u 0‖ ≤ C * b 0 + C * ‖h‖ := by
      have :=
        calc
          ‖v 0‖ ≤ ‖h‖ + ‖v 0 - h‖ := norm_le_insert' _ _
          _ ≤ ‖h‖ + b 0 := by gcongr
      calc
        ‖u 0‖ ≤ C * ‖v 0‖ := hnorm_u 0
        _ ≤ C * (‖h‖ + b 0) := by gcongr
        _ = C * b 0 + C * ‖h‖ := by rw [add_comm, mul_add]
    have : (∑ k ∈ range (n + 1), C * b k) ≤ ε * ‖h‖ :=
      calc (∑ k ∈ range (n + 1), C * b k)
        _ = (∑ k ∈ range (n + 1), (1 / 2 : ℝ) ^ k) * (ε * ‖h‖ / 2) := by
          simp only [b, mul_div_cancel₀ _ hC.ne.symm, ← sum_mul]
        _ ≤ 2 * (ε * ‖h‖ / 2) := by gcongr; apply sum_geometric_two_le
        _ = ε * ‖h‖ := mul_div_cancel₀ _ two_ne_zero
    calc
      ‖s n‖ ≤ ∑ k ∈ range (n + 1), ‖u k‖ := norm_sum_le _ _
      _ = (∑ k ∈ range n, ‖u (k + 1)‖) + ‖u 0‖ := sum_range_succ' _ _
      _ ≤ (∑ k ∈ range n, C * ‖v (k + 1)‖) + ‖u 0‖ := by gcongr; apply hnorm_u
      _ ≤ (∑ k ∈ range n, C * b (k + 1)) + (C * b 0 + C * ‖h‖) := by
        gcongr with k; exact (hv _ k.succ_pos).le
      _ = (∑ k ∈ range (n + 1), C * b k) + C * ‖h‖ := by rw [← add_assoc, sum_range_succ']
      _ ≤ (C + ε) * ‖h‖ := by
        rw [add_comm, add_mul]
        apply add_le_add_left this


/-- Given `f : NormedAddGroupHom G H` for some complete `G`, if every element `x` of the image of
an isometric immersion `j : NormedAddGroupHom K H` has a preimage under `f` whose norm is at most
`C*‖x‖` then the same holds for elements of the (topological) closure of this image with constant
`C+ε` instead of `C`, for any positive `ε`.
This is useful in particular if `j` is the inclusion of a normed group into its completion
(in this case the closure is the full target group).
-/
theorem controlled_closure_range_of_complete {f : NormedAddGroupHom G H} {K : Type*}
    [SeminormedAddCommGroup K] {j : NormedAddGroupHom K H} (hj : ∀ x, ‖j x‖ = ‖x‖) {C ε : ℝ}
    (hC : 0 < C) (hε : 0 < ε) (hyp : ∀ k, ∃ g, f g = j k ∧ ‖g‖ ≤ C * ‖k‖) :
    f.SurjectiveOnWith j.range.topologicalClosure (C + ε) := by
  replace hyp : ∀ h ∈ j.range, ∃ g, f g = h ∧ ‖g‖ ≤ C * ‖h‖ := by
    intro h h_in
    rcases (j.mem_range _).mp h_in with ⟨k, rfl⟩
    rw [hj]
    exact hyp k
  /-
    G : Type u_1
    inst✝³ : NormedAddCommGroup G
    inst✝² : CompleteSpace G
    H : Type u_2
    inst✝¹ : NormedAddCommGroup H
    f : NormedAddGroupHom G H
    K : Type u_3
    inst✝ : SeminormedAddCommGroup K
    j : NormedAddGroupHom K H
    hj : ∀ (x : K), Eq (Norm.norm (j x)) (Norm.norm x)
    C ε : Real
    hC : LT.lt 0 C
    hε : LT.lt 0 ε
    hyp : ∀ (h : H), Membership.mem j.range h → Exists fun g => And (Eq (f g) h) ( …
    ⊢ f.SurjectiveOnWith j.range.topologicalClosure (HAdd.hAdd C ε)
  -/
  exact controlled_closure_of_complete hC hε hyp
  /-
    🎉 no goals
  -/

