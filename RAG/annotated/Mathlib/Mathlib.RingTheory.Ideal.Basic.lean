/-- `I^n` as an ideal of `R^n`. -/
def pi : Ideal (ι → α) where
  carrier := { x | ∀ i, x i ∈ I }
  zero_mem' _i := I.zero_mem
  add_mem' ha hb i := I.add_mem (ha i) (hb i)
  smul_mem' a _b hb i := I.mul_mem_left (a i) (hb i)


theorem mem_pi (x : ι → α) : x ∈ I.pi ι ↔ ∀ i, x i ∈ I :=
  Iff.rfl


theorem add_pow_mem_of_pow_mem_of_le {m n k : ℕ}
    (ha : a ^ m ∈ I) (hb : b ^ n ∈ I) (hk : m + n ≤ k + 1) :
    (a + b) ^ k ∈ I := by
  /-
    α : Type u
    a b : α
    inst✝ : CommSemiring α
    I : Ideal α
    m n k : Nat
    ha : Membership.mem I (HPow.hPow a m)
    hb : Membership.mem I (HPow.hPow b n)
    hk : LE.le (HAdd.hAdd m n) (HAdd.hAdd k 1)
    ⊢ Membership.mem I (HPow.hPow (HAdd.hAdd a b) k)
  -/
  rw [add_pow]
  /-
    α : Type u
    a b : α
    inst✝ : CommSemiring α
    I : Ideal α
    m n k : Nat
    ha : Membership.mem I (HPow.hPow a m)
    hb : Membership.mem I (HPow.hPow b n)
    hk : LE.le (HAdd.hAdd m n) (HAdd.hAdd k 1)
    ⊢ Membership.mem I ((Finset.range (HAdd.hAdd k 1)).sum fun m => HMul.hMul (HMu …
  -/
  apply I.sum_mem
  /-
    α : Type u
    a b : α
    inst✝ : CommSemiring α
    I : Ideal α
    m n k : Nat
    ha : Membership.mem I (HPow.hPow a m)
    hb : Membership.mem I (HPow.hPow b n)
    hk : LE.le (HAdd.hAdd m n) (HAdd.hAdd k 1)
    ⊢ ∀ (c : Nat), Membership.mem (Finset.range (HAdd.hAdd k 1)) c → Membership.me …
  -/
  intro c _
  /-
    α : Type u
    a b : α
    inst✝ : CommSemiring α
    I : Ideal α
    m n k : Nat
    ha : Membership.mem I (HPow.hPow a m)
    hb : Membership.mem I (HPow.hPow b n)
    hk : LE.le (HAdd.hAdd m n) (HAdd.hAdd k 1)
    c : Nat
    a✝ : Membership.mem (Finset.range (HAdd.hAdd k 1)) c
    ⊢ Membership.mem I (HMul.hMul (HMul.hMul (HPow.hPow a c) (HPow.hPow b (HSub.hS …
  -/
  apply mul_mem_right
  /-
    case h
    α : Type u
    a b : α
    inst✝ : CommSemiring α
    I : Ideal α
    m n k : Nat
    ha : Membership.mem I (HPow.hPow a m)
    hb : Membership.mem I (HPow.hPow b n)
    hk : LE.le (HAdd.hAdd m n) (HAdd.hAdd k 1)
    c : Nat
    a✝ : Membership.mem (Finset.range (HAdd.hAdd k 1)) c
    ⊢ Membership.mem I (HMul.hMul (HPow.hPow a c) (HPow.hPow b (HSub.hSub k c)))
  -/
  by_cases h : m ≤ c
    /-
      case pos
      α : Type u
      a b : α
      inst✝ : CommSemiring α
      I : Ideal α
      m n k : Nat
      ha : Membership.mem I (HPow.hPow a m)
      hb : Membership.mem I (HPow.hPow b n)
      hk : LE.le (HAdd.hAdd m n) (HAdd.hAdd k 1)
      c : Nat
      a✝ : Membership.mem (Finset.range (HAdd.hAdd k 1)) c
      h : LE.le m c
      ⊢ Membership.mem I (HMul.hMul (HPow.hPow a c) (HPow.hPow b (HSub.hSub k c)))
    -/
  · exact I.mul_mem_right _ (I.pow_mem_of_pow_mem ha h)
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u
      a b : α
      inst✝ : CommSemiring α
      I : Ideal α
      m n k : Nat
      ha : Membership.mem I (HPow.hPow a m)
      hb : Membership.mem I (HPow.hPow b n)
      hk : LE.le (HAdd.hAdd m n) (HAdd.hAdd k 1)
      c : Nat
      a✝ : Membership.mem (Finset.range (HAdd.hAdd k 1)) c
      h : Not (LE.le m c)
      ⊢ Membership.mem I (HMul.hMul (HPow.hPow a c) (HPow.hPow b (HSub.hSub k c)))
    -/
  · refine I.mul_mem_left _ (I.pow_mem_of_pow_mem hb ?_)
    /-
      case neg
      α : Type u
      a b : α
      inst✝ : CommSemiring α
      I : Ideal α
      m n k : Nat
      ha : Membership.mem I (HPow.hPow a m)
      hb : Membership.mem I (HPow.hPow b n)
      hk : LE.le (HAdd.hAdd m n) (HAdd.hAdd k 1)
      c : Nat
      a✝ : Membership.mem (Finset.range (HAdd.hAdd k 1)) c
      h : Not (LE.le m c)
      ⊢ LE.le n (HSub.hSub k c)
    -/
    simp only [not_le, Nat.lt_iff_add_one_le] at h
    have hck : c ≤ k := by
      rw [← add_le_add_iff_right 1]
      exact le_trans h (le_trans (Nat.le_add_right _ _) hk)
    /-
      case neg
      α : Type u
      a b : α
      inst✝ : CommSemiring α
      I : Ideal α
      m n k : Nat
      ha : Membership.mem I (HPow.hPow a m)
      hb : Membership.mem I (HPow.hPow b n)
      hk : LE.le (HAdd.hAdd m n) (HAdd.hAdd k 1)
      c : Nat
      a✝ : Membership.mem (Finset.range (HAdd.hAdd k 1)) c
      h : LE.le (HAdd.hAdd c 1) m
      hck : LE.le c k
      ⊢ LE.le n (HSub.hSub k c)
    -/
    rw [Nat.le_sub_iff_add_le hck, ← add_le_add_iff_right 1]
    /-
      case neg
      α : Type u
      a b : α
      inst✝ : CommSemiring α
      I : Ideal α
      m n k : Nat
      ha : Membership.mem I (HPow.hPow a m)
      hb : Membership.mem I (HPow.hPow b n)
      hk : LE.le (HAdd.hAdd m n) (HAdd.hAdd k 1)
      c : Nat
      a✝ : Membership.mem (Finset.range (HAdd.hAdd k 1)) c
      h : LE.le (HAdd.hAdd c 1) m
      hck : LE.le c k
      ⊢ LE.le (HAdd.hAdd (HAdd.hAdd n c) 1) (HAdd.hAdd k 1)
    -/
    exact le_trans (by rwa [add_comm _ n, add_assoc, add_le_add_iff_left]) hk
    /-
      🎉 no goals
    -/


theorem add_pow_add_pred_mem_of_pow_mem  {m n : ℕ}
    (ha : a ^ m ∈ I) (hb : b ^ n ∈ I) :
    (a + b) ^ (m + n - 1) ∈ I :=
                                             /-
                                               α : Type u
                                               a b : α
                                               inst✝ : CommSemiring α
                                               I : Ideal α
                                               m n : Nat
                                               ha : Membership.mem I (HPow.hPow a m)
                                               hb : Membership.mem I (HPow.hPow b n)
                                               ⊢ LE.le (HAdd.hAdd m n) (HAdd.hAdd (HSub.hSub (HAdd.hAdd m n) 1) 1)
                                             -/
  I.add_pow_mem_of_pow_mem_of_le ha hb <| by rw [← Nat.sub_le_iff_le_add]
                                             /-
                                               🎉 no goals
                                             -/


theorem pow_multiset_sum_mem_span_pow [DecidableEq α] (s : Multiset α) (n : ℕ) :
    s.sum ^ (Multiset.card s * n + 1) ∈
    span ((s.map fun (x : α) ↦ x ^ (n + 1)).toFinset : Set α) := by
  /-
    α : Type u
    inst✝¹ : CommSemiring α
    inst✝ : DecidableEq α
    s : Multiset α
    n : Nat
    ⊢ Membership.mem (Ideal.span ↑(Multiset.map (fun x => HPow.hPow x (HAdd.hAdd n …
  -/
  induction' s using Multiset.induction_on with a s hs
    /-
      case empty
      α : Type u
      inst✝¹ : CommSemiring α
      inst✝ : DecidableEq α
      n : Nat
      ⊢ Membership.mem (Ideal.span ↑(Multiset.map (fun x => HPow.hPow x (HAdd.hAdd n …
    -/
  · simp
    /-
      🎉 no goals
    -/
  simp only [Finset.coe_insert, Multiset.map_cons, Multiset.toFinset_cons, Multiset.sum_cons,
    Multiset.card_cons, add_pow]
  /-
    case cons
    α : Type u
    inst✝¹ : CommSemiring α
    inst✝ : DecidableEq α
    n : Nat
    a : α
    s : Multiset α
    hs : Membership.mem (Ideal.span ↑(Multiset.map (fun x => HPow.hPow x (HAdd.hAd …
    ⊢ Membership.mem (Ideal.span (Insert.insert (HPow.hPow a (HAdd.hAdd n 1)) ↑(Mu …
  -/
  refine Submodule.sum_mem _ ?_
  /-
    case cons
    α : Type u
    inst✝¹ : CommSemiring α
    inst✝ : DecidableEq α
    n : Nat
    a : α
    s : Multiset α
    hs : Membership.mem (Ideal.span ↑(Multiset.map (fun x => HPow.hPow x (HAdd.hAd …
    ⊢ ∀ (c : Nat), Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd (HMul.hMul ( …
  -/
  intro c _hc
  /-
    case cons
    α : Type u
    inst✝¹ : CommSemiring α
    inst✝ : DecidableEq α
    n : Nat
    a : α
    s : Multiset α
    hs : Membership.mem (Ideal.span ↑(Multiset.map (fun x => HPow.hPow x (HAdd.hAd …
    c : Nat
    _hc : Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HAdd.hAdd …
    ⊢ Membership.mem (Ideal.span (Insert.insert (HPow.hPow a (HAdd.hAdd n 1)) ↑(Mu …
  -/
  rw [mem_span_insert]
  /-
    case cons
    α : Type u
    inst✝¹ : CommSemiring α
    inst✝ : DecidableEq α
    n : Nat
    a : α
    s : Multiset α
    hs : Membership.mem (Ideal.span ↑(Multiset.map (fun x => HPow.hPow x (HAdd.hAd …
    c : Nat
    _hc : Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HAdd.hAdd …
    ⊢ Exists fun a_1 => Exists fun z => And (Membership.mem (Ideal.span ↑(Multiset …
  -/
  by_cases h : n + 1 ≤ c
  · refine ⟨a ^ (c - (n + 1)) * s.sum ^ ((Multiset.card s + 1) * n + 1 - c) *
      ((Multiset.card s + 1) * n + 1).choose c, 0, Submodule.zero_mem _, ?_⟩
    /-
      case pos
      α : Type u
      inst✝¹ : CommSemiring α
      inst✝ : DecidableEq α
      n : Nat
      a : α
      s : Multiset α
      hs : Membership.mem (Ideal.span ↑(Multiset.map (fun x => HPow.hPow x (HAdd.hAd …
      c : Nat
      _hc : Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HAdd.hAdd …
      h : LE.le (HAdd.hAdd n 1) c
      ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow a c) (HPow.hPow s.sum (HSub.hSub (HAdd.h …
    -/
    rw [mul_comm _ (a ^ (n + 1))]
    /-
      case pos
      α : Type u
      inst✝¹ : CommSemiring α
      inst✝ : DecidableEq α
      n : Nat
      a : α
      s : Multiset α
      hs : Membership.mem (Ideal.span ↑(Multiset.map (fun x => HPow.hPow x (HAdd.hAd …
      c : Nat
      _hc : Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HAdd.hAdd …
      h : LE.le (HAdd.hAdd n 1) c
      ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow a c) (HPow.hPow s.sum (HSub.hSub (HAdd.h …
    -/
    simp_rw [← mul_assoc]
    /-
      case pos
      α : Type u
      inst✝¹ : CommSemiring α
      inst✝ : DecidableEq α
      n : Nat
      a : α
      s : Multiset α
      hs : Membership.mem (Ideal.span ↑(Multiset.map (fun x => HPow.hPow x (HAdd.hAd …
      c : Nat
      _hc : Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HAdd.hAdd …
      h : LE.le (HAdd.hAdd n 1) c
      ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow a c) (HPow.hPow s.sum (HSub.hSub (HAdd.h …
    -/
    rw [← pow_add, add_zero, add_tsub_cancel_of_le h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u
      inst✝¹ : CommSemiring α
      inst✝ : DecidableEq α
      n : Nat
      a : α
      s : Multiset α
      hs : Membership.mem (Ideal.span ↑(Multiset.map (fun x => HPow.hPow x (HAdd.hAd …
      c : Nat
      _hc : Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HAdd.hAdd …
      h : Not (LE.le (HAdd.hAdd n 1) c)
      ⊢ Exists fun a_1 => Exists fun z => And (Membership.mem (Ideal.span ↑(Multiset …
    -/
  · use 0
    /-
      case h
      α : Type u
      inst✝¹ : CommSemiring α
      inst✝ : DecidableEq α
      n : Nat
      a : α
      s : Multiset α
      hs : Membership.mem (Ideal.span ↑(Multiset.map (fun x => HPow.hPow x (HAdd.hAd …
      c : Nat
      _hc : Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HAdd.hAdd …
      h : Not (LE.le (HAdd.hAdd n 1) c)
      ⊢ Exists fun z => And (Membership.mem (Ideal.span ↑(Multiset.map (fun x => HPo …
    -/
    simp_rw [zero_mul, zero_add]
    /-
      case h
      α : Type u
      inst✝¹ : CommSemiring α
      inst✝ : DecidableEq α
      n : Nat
      a : α
      s : Multiset α
      hs : Membership.mem (Ideal.span ↑(Multiset.map (fun x => HPow.hPow x (HAdd.hAd …
      c : Nat
      _hc : Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HAdd.hAdd …
      h : Not (LE.le (HAdd.hAdd n 1) c)
      ⊢ Exists fun z => And (Membership.mem (Ideal.span ↑(Multiset.map (fun x => HPo …
    -/
    refine ⟨_, ?_, rfl⟩
    /-
      case h
      α : Type u
      inst✝¹ : CommSemiring α
      inst✝ : DecidableEq α
      n : Nat
      a : α
      s : Multiset α
      hs : Membership.mem (Ideal.span ↑(Multiset.map (fun x => HPow.hPow x (HAdd.hAd …
      c : Nat
      _hc : Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HAdd.hAdd …
      h : Not (LE.le (HAdd.hAdd n 1) c)
      ⊢ Membership.mem (Ideal.span ↑(Multiset.map (fun x => HPow.hPow x (HAdd.hAdd n …
    -/
    replace h : c ≤ n := Nat.lt_succ_iff.mp (not_le.mp h)
    have : (Multiset.card s + 1) * n + 1 - c = Multiset.card s * n + 1 + (n - c) := by
      rw [add_mul, one_mul, add_assoc, add_comm n 1, ← add_assoc, add_tsub_assoc_of_le h]
    /-
      case h
      α : Type u
      inst✝¹ : CommSemiring α
      inst✝ : DecidableEq α
      n : Nat
      a : α
      s : Multiset α
      hs : Membership.mem (Ideal.span ↑(Multiset.map (fun x => HPow.hPow x (HAdd.hAd …
      c : Nat
      _hc : Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HAdd.hAdd …
      h : LE.le c n
      this : Eq (HSub.hSub (HAdd.hAdd (HMul.hMul (HAdd.hAdd s.card 1) n) 1) c) (HAdd …
      ⊢ Membership.mem (Ideal.span ↑(Multiset.map (fun x => HPow.hPow x (HAdd.hAdd n …
    -/
    rw [this, pow_add]
    /-
      case h
      α : Type u
      inst✝¹ : CommSemiring α
      inst✝ : DecidableEq α
      n : Nat
      a : α
      s : Multiset α
      hs : Membership.mem (Ideal.span ↑(Multiset.map (fun x => HPow.hPow x (HAdd.hAd …
      c : Nat
      _hc : Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HAdd.hAdd …
      h : LE.le c n
      this : Eq (HSub.hSub (HAdd.hAdd (HMul.hMul (HAdd.hAdd s.card 1) n) 1) c) (HAdd …
      ⊢ Membership.mem (Ideal.span ↑(Multiset.map (fun x => HPow.hPow x (HAdd.hAdd n …
    -/
    simp_rw [mul_assoc, mul_comm (s.sum ^ (Multiset.card s * n + 1)), ← mul_assoc]
    /-
      case h
      α : Type u
      inst✝¹ : CommSemiring α
      inst✝ : DecidableEq α
      n : Nat
      a : α
      s : Multiset α
      hs : Membership.mem (Ideal.span ↑(Multiset.map (fun x => HPow.hPow x (HAdd.hAd …
      c : Nat
      _hc : Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HAdd.hAdd …
      h : LE.le c n
      this : Eq (HSub.hSub (HAdd.hAdd (HMul.hMul (HAdd.hAdd s.card 1) n) 1) c) (HAdd …
      ⊢ Membership.mem (Ideal.span ↑(Multiset.map (fun x => HPow.hPow x (HAdd.hAdd n …
    -/
    exact mul_mem_left _ _ hs
    /-
      🎉 no goals
    -/


theorem sum_pow_mem_span_pow {ι} (s : Finset ι) (f : ι → α) (n : ℕ) :
    (∑ i ∈ s, f i) ^ (s.card * n + 1) ∈ span ((fun i => f i ^ (n + 1)) '' s) := by
  classical
  simpa only [Multiset.card_map, Multiset.map_map, comp_apply, Multiset.toFinset_map,
    Finset.coe_image, Finset.val_toFinset] using pow_multiset_sum_mem_span_pow (s.1.map f) n


theorem span_pow_eq_top (s : Set α) (hs : span s = ⊤) (n : ℕ) :
    span ((fun (x : α) => x ^ n) '' s) = ⊤ := by
  /-
    α : Type u
    inst✝ : CommSemiring α
    s : Set α
    hs : Eq (Ideal.span s) Top.top
    n : Nat
    ⊢ Eq (Ideal.span (Set.image (fun x => HPow.hPow x n) s)) Top.top
  -/
  rw [eq_top_iff_one]
  /-
    α : Type u
    inst✝ : CommSemiring α
    s : Set α
    hs : Eq (Ideal.span s) Top.top
    n : Nat
    ⊢ Membership.mem (Ideal.span (Set.image (fun x => HPow.hPow x n) s)) 1
  -/
  cases' n with n
    /-
      case zero
      α : Type u
      inst✝ : CommSemiring α
      s : Set α
      hs : Eq (Ideal.span s) Top.top
      ⊢ Membership.mem (Ideal.span (Set.image (fun x => HPow.hPow x 0) s)) 1
    -/
  · obtain rfl | ⟨x, hx⟩ := eq_empty_or_nonempty s
      /-
        case zero.inl
        α : Type u
        inst✝ : CommSemiring α
        hs : Eq (Ideal.span EmptyCollection.emptyCollection) Top.top
        ⊢ Membership.mem (Ideal.span (Set.image (fun x => HPow.hPow x 0) EmptyCollecti …
      -/
    · rw [Set.image_empty, hs]
      /-
        case zero.inl
        α : Type u
        inst✝ : CommSemiring α
        hs : Eq (Ideal.span EmptyCollection.emptyCollection) Top.top
        ⊢ Membership.mem Top.top 1
      -/
      trivial
      /-
        🎉 no goals
      -/
      /-
        case zero.inr.intro
        α : Type u
        inst✝ : CommSemiring α
        s : Set α
        hs : Eq (Ideal.span s) Top.top
        x : α
        hx : Membership.mem s x
        ⊢ Membership.mem (Ideal.span (Set.image (fun x => HPow.hPow x 0) s)) 1
      -/
    · exact subset_span ⟨_, hx, pow_zero _⟩
      /-
        🎉 no goals
      -/
  /-
    case succ
    α : Type u
    inst✝ : CommSemiring α
    s : Set α
    hs : Eq (Ideal.span s) Top.top
    n : Nat
    ⊢ Membership.mem (Ideal.span (Set.image (fun x => HPow.hPow x (HAdd.hAdd n 1)) …
  -/
  rw [eq_top_iff_one, span, Finsupp.mem_span_iff_linearCombination] at hs
  /-
    case succ
    α : Type u
    inst✝ : CommSemiring α
    s : Set α
    hs : Exists fun l => Eq ((Finsupp.linearCombination α Subtype.val) l) 1
    n : Nat
    ⊢ Membership.mem (Ideal.span (Set.image (fun x => HPow.hPow x (HAdd.hAdd n 1)) …
  -/
  rcases hs with ⟨f, hf⟩
  /-
    case succ.intro
    α : Type u
    inst✝ : CommSemiring α
    s : Set α
    n : Nat
    f : Finsupp (↑s) α
    hf : Eq ((Finsupp.linearCombination α Subtype.val) f) 1
    ⊢ Membership.mem (Ideal.span (Set.image (fun x => HPow.hPow x (HAdd.hAdd n 1)) …
  -/
  have hf : (f.support.sum fun a => f a * a) = 1 := hf -- Porting note: was `change ... at hf`
  /-
    case succ.intro
    α : Type u
    inst✝ : CommSemiring α
    s : Set α
    n : Nat
    f : Finsupp (↑s) α
    hf✝ : Eq ((Finsupp.linearCombination α Subtype.val) f) 1
    hf : Eq (f.support.sum fun a => HMul.hMul (f a) ↑a) 1
    ⊢ Membership.mem (Ideal.span (Set.image (fun x => HPow.hPow x (HAdd.hAdd n 1)) …
  -/
  have := sum_pow_mem_span_pow f.support (fun a => f a * a) n
  /-
    case succ.intro
    α : Type u
    inst✝ : CommSemiring α
    s : Set α
    n : Nat
    f : Finsupp (↑s) α
    hf✝ : Eq ((Finsupp.linearCombination α Subtype.val) f) 1
    hf : Eq (f.support.sum fun a => HMul.hMul (f a) ↑a) 1
    this : Membership.mem (Ideal.span (Set.image (fun i => HPow.hPow (HMul.hMul (f …
    ⊢ Membership.mem (Ideal.span (Set.image (fun x => HPow.hPow x (HAdd.hAdd n 1)) …
  -/
  rw [hf, one_pow] at this
  /-
    case succ.intro
    α : Type u
    inst✝ : CommSemiring α
    s : Set α
    n : Nat
    f : Finsupp (↑s) α
    hf✝ : Eq ((Finsupp.linearCombination α Subtype.val) f) 1
    hf : Eq (f.support.sum fun a => HMul.hMul (f a) ↑a) 1
    this : Membership.mem (Ideal.span (Set.image (fun i => HPow.hPow (HMul.hMul (f …
    ⊢ Membership.mem (Ideal.span (Set.image (fun x => HPow.hPow x (HAdd.hAdd n 1)) …
  -/
  refine span_le.mpr ?_ this
  /-
    case succ.intro
    α : Type u
    inst✝ : CommSemiring α
    s : Set α
    n : Nat
    f : Finsupp (↑s) α
    hf✝ : Eq ((Finsupp.linearCombination α Subtype.val) f) 1
    hf : Eq (f.support.sum fun a => HMul.hMul (f a) ↑a) 1
    this : Membership.mem (Ideal.span (Set.image (fun i => HPow.hPow (HMul.hMul (f …
    ⊢ HasSubset.Subset (Set.image (fun i => HPow.hPow (HMul.hMul (f i) ↑i) (HAdd.h …
  -/
  rintro _ hx
  /-
    case succ.intro
    α : Type u
    inst✝ : CommSemiring α
    s : Set α
    n : Nat
    f : Finsupp (↑s) α
    hf✝ : Eq ((Finsupp.linearCombination α Subtype.val) f) 1
    hf : Eq (f.support.sum fun a => HMul.hMul (f a) ↑a) 1
    this : Membership.mem (Ideal.span (Set.image (fun i => HPow.hPow (HMul.hMul (f …
    a✝ : α
    hx : Membership.mem (Set.image (fun i => HPow.hPow (HMul.hMul (f i) ↑i) (HAdd. …
    ⊢ Membership.mem (↑(Ideal.span (Set.image (fun x => HPow.hPow x (HAdd.hAdd n 1 …
  -/
  simp_rw [Set.mem_image] at hx
  /-
    case succ.intro
    α : Type u
    inst✝ : CommSemiring α
    s : Set α
    n : Nat
    f : Finsupp (↑s) α
    hf✝ : Eq ((Finsupp.linearCombination α Subtype.val) f) 1
    hf : Eq (f.support.sum fun a => HMul.hMul (f a) ↑a) 1
    this : Membership.mem (Ideal.span (Set.image (fun i => HPow.hPow (HMul.hMul (f …
    a✝ : α
    hx : Exists fun x => And (Membership.mem (↑f.support) x) (Eq (HPow.hPow (HMul. …
    ⊢ Membership.mem (↑(Ideal.span (Set.image (fun x => HPow.hPow x (HAdd.hAdd n 1 …
  -/
  rcases hx with ⟨x, _, rfl⟩
  have : span ({(x : α) ^ (n + 1)} : Set α) ≤ span ((fun x : α => x ^ (n + 1)) '' s) := by
    rw [span_le, Set.singleton_subset_iff]
    exact subset_span ⟨x, x.prop, rfl⟩
  /-
    case succ.intro.intro.intro
    α : Type u
    inst✝ : CommSemiring α
    s : Set α
    n : Nat
    f : Finsupp (↑s) α
    hf✝ : Eq ((Finsupp.linearCombination α Subtype.val) f) 1
    hf : Eq (f.support.sum fun a => HMul.hMul (f a) ↑a) 1
    this✝ : Membership.mem (Ideal.span (Set.image (fun i => HPow.hPow (HMul.hMul ( …
    x : ↑s
    left✝ : Membership.mem (↑f.support) x
    this : LE.le (Ideal.span (Singleton.singleton (HPow.hPow (↑x) (HAdd.hAdd n 1)) …
    ⊢ Membership.mem (↑(Ideal.span (Set.image (fun x => HPow.hPow x (HAdd.hAdd n 1 …
  -/
  refine this ?_
  /-
    case succ.intro.intro.intro
    α : Type u
    inst✝ : CommSemiring α
    s : Set α
    n : Nat
    f : Finsupp (↑s) α
    hf✝ : Eq ((Finsupp.linearCombination α Subtype.val) f) 1
    hf : Eq (f.support.sum fun a => HMul.hMul (f a) ↑a) 1
    this✝ : Membership.mem (Ideal.span (Set.image (fun i => HPow.hPow (HMul.hMul ( …
    x : ↑s
    left✝ : Membership.mem (↑f.support) x
    this : LE.le (Ideal.span (Singleton.singleton (HPow.hPow (↑x) (HAdd.hAdd n 1)) …
    ⊢ Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑x) (HAdd.hAdd n …
  -/
  rw [mul_pow, mem_span_singleton]
  /-
    case succ.intro.intro.intro
    α : Type u
    inst✝ : CommSemiring α
    s : Set α
    n : Nat
    f : Finsupp (↑s) α
    hf✝ : Eq ((Finsupp.linearCombination α Subtype.val) f) 1
    hf : Eq (f.support.sum fun a => HMul.hMul (f a) ↑a) 1
    this✝ : Membership.mem (Ideal.span (Set.image (fun i => HPow.hPow (HMul.hMul ( …
    x : ↑s
    left✝ : Membership.mem (↑f.support) x
    this : LE.le (Ideal.span (Singleton.singleton (HPow.hPow (↑x) (HAdd.hAdd n 1)) …
    ⊢ Dvd.dvd (HPow.hPow (↑x) (HAdd.hAdd n 1)) (HMul.hMul (HPow.hPow (f x) (HAdd.h …
  -/
  exact ⟨f x ^ (n + 1), mul_comm _ _⟩
  /-
    🎉 no goals
  -/


theorem span_range_pow_eq_top (s : Set α) (hs : span s = ⊤) (n : s → ℕ) :
    span (Set.range fun x ↦ x.1 ^ n x) = ⊤ := by
  /-
    α : Type u
    inst✝ : CommSemiring α
    s : Set α
    hs : Eq (Ideal.span s) Top.top
    n : ↑s → Nat
    ⊢ Eq (Ideal.span (Set.range fun x => HPow.hPow (↑x) (n x))) Top.top
  -/
  have ⟨t, hts, mem⟩ := Submodule.mem_span_finite_of_mem_span ((eq_top_iff_one _).mp hs)
  refine top_unique ((span_pow_eq_top _ ((eq_top_iff_one _).mpr mem) <|
    t.attach.sup fun x ↦ n ⟨x, hts x.2⟩).ge.trans <| span_le.mpr ?_)
  /-
    α : Type u
    inst✝ : CommSemiring α
    s : Set α
    hs : Eq (Ideal.span s) Top.top
    n : ↑s → Nat
    t : Finset α
    hts : HasSubset.Subset (↑t) s
    mem : Membership.mem (Submodule.span α ↑t) 1
    ⊢ HasSubset.Subset (Set.image (fun x => HPow.hPow x (t.attach.sup fun x => n ⟨ …
  -/
  rintro _ ⟨x, hxt, rfl⟩
  /-
    case intro.intro
    α : Type u
    inst✝ : CommSemiring α
    s : Set α
    hs : Eq (Ideal.span s) Top.top
    n : ↑s → Nat
    t : Finset α
    hts : HasSubset.Subset (↑t) s
    mem : Membership.mem (Submodule.span α ↑t) 1
    x : α
    hxt : Membership.mem (↑t) x
    ⊢ Membership.mem (↑(Ideal.span (Set.range fun x => HPow.hPow (↑x) (n x)))) ((f …
  -/
  rw [← Nat.sub_add_cancel (Finset.le_sup <| t.mem_attach ⟨x, hxt⟩)]
  /-
    case intro.intro
    α : Type u
    inst✝ : CommSemiring α
    s : Set α
    hs : Eq (Ideal.span s) Top.top
    n : ↑s → Nat
    t : Finset α
    hts : HasSubset.Subset (↑t) s
    mem : Membership.mem (Submodule.span α ↑t) 1
    x : α
    hxt : Membership.mem (↑t) x
    ⊢ Membership.mem (↑(Ideal.span (Set.range fun x => HPow.hPow (↑x) (n x)))) ((f …
  -/
  simp_rw [pow_add]
  /-
    case intro.intro
    α : Type u
    inst✝ : CommSemiring α
    s : Set α
    hs : Eq (Ideal.span s) Top.top
    n : ↑s → Nat
    t : Finset α
    hts : HasSubset.Subset (↑t) s
    mem : Membership.mem (Submodule.span α ↑t) 1
    x : α
    hxt : Membership.mem (↑t) x
    ⊢ Membership.mem (↑(Ideal.span (Set.range fun x => HPow.hPow (↑x) (n x)))) (HM …
  -/
  exact mul_mem_left _ _ (subset_span ⟨_, rfl⟩)
  /-
    🎉 no goals
  -/


variable (K) in
/-- A bijection between (left) ideals of a division ring and `{0, 1}`, sending `⊥` to `0`
and `⊤` to `1`. -/
def equivFinTwo [DecidableEq (Ideal K)] : Ideal K ≃ Fin 2 where
  toFun := fun I ↦ if I = ⊥ then 0 else 1
  invFun := ![⊥, ⊤]
                         /-
                           α : Type u
                           β : Type v
                           F : Type w
                           K : Type u
                           inst✝¹ : DivisionSemiring K
                           I✝ : Ideal K
                           inst✝ : DecidableEq (Ideal K)
                           I : Ideal K
                           ⊢ Eq (Matrix.vecCons Bot.bot (Matrix.vecCons Top.top Matrix.vecEmpty) ((fun I  …
                         -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  left_inv := fun I ↦ by rcases eq_bot_or_top I with rfl | rfl <;> simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
                          /-
                            α : Type u
                            β : Type v
                            F : Type w
                            K : Type u
                            inst✝¹ : DivisionSemiring K
                            I : Ideal K
                            inst✝ : DecidableEq (Ideal K)
                            i : Fin 2
                            ⊢ Eq ((fun I => ite (Eq I Bot.bot) 0 1) (Matrix.vecCons Bot.bot (Matrix.vecCon …
                          -/
                                          /-
                                            🎉 no goals
                                          -/
  right_inv := fun i ↦ by fin_cases i <;> simp
                                          /-
                                            🎉 no goals
                                          -/


instance : Finite (Ideal K) := let _i := Classical.decEq (Ideal K); ⟨equivFinTwo K⟩


/-- Ideals of a `DivisionSemiring` are a simple order. Thanks to the way abbreviations work,
this automatically gives an `IsSimpleModule K` instance. -/
instance isSimpleOrder : IsSimpleOrder (Ideal K) :=
  ⟨eq_bot_or_top⟩


theorem exists_not_isUnit_of_not_isField [Nontrivial R] (hf : ¬IsField R) :
    ∃ (x : R) (_hx : x ≠ (0 : R)), ¬IsUnit x := by
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    hf : Not (IsField R)
    ⊢ Exists fun x => Exists fun _hx => Not (IsUnit x)
  -/
  have : ¬_ := fun h => hf ⟨exists_pair_ne R, mul_comm, h⟩
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    hf : Not (IsField R)
    this : Not (∀ {a : R}, Ne a 0 → Exists fun b => Eq (HMul.hMul a b) 1)
    ⊢ Exists fun x => Exists fun _hx => Not (IsUnit x)
  -/
  simp_rw [isUnit_iff_exists_inv]
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    hf : Not (IsField R)
    this : Not (∀ {a : R}, Ne a 0 → Exists fun b => Eq (HMul.hMul a b) 1)
    ⊢ Exists fun x => Exists fun h => Not (Exists fun b => Eq (HMul.hMul x b) 1)
  -/
  push_neg at this ⊢
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    hf : Not (IsField R)
    this : Exists fun {a} => And (Ne a 0) (∀ (b : R), Ne (HMul.hMul a b) 1)
    ⊢ Exists fun x => Exists fun h => ∀ (b : R), Ne (HMul.hMul x b) 1
  -/
  obtain ⟨x, hx, not_unit⟩ := this
  /-
    case intro.intro
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    hf : Not (IsField R)
    x : R
    hx : Ne x 0
    not_unit : ∀ (b : R), Ne (HMul.hMul x b) 1
    ⊢ Exists fun x => Exists fun h => ∀ (b : R), Ne (HMul.hMul x b) 1
  -/
  exact ⟨x, hx, not_unit⟩
  /-
    🎉 no goals
  -/


theorem not_isField_iff_exists_ideal_bot_lt_and_lt_top [Nontrivial R] :
    ¬IsField R ↔ ∃ I : Ideal R, ⊥ < I ∧ I < ⊤ := by
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    ⊢ Iff (Not (IsField R)) (Exists fun I => And (LT.lt Bot.bot I) (LT.lt I Top.to …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : Nontrivial R
      ⊢ Not (IsField R) → Exists fun I => And (LT.lt Bot.bot I) (LT.lt I Top.top)
    -/
  · intro h
    /-
      case mp
      R : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : Nontrivial R
      h : Not (IsField R)
      ⊢ Exists fun I => And (LT.lt Bot.bot I) (LT.lt I Top.top)
    -/
    obtain ⟨x, nz, nu⟩ := exists_not_isUnit_of_not_isField h
    /-
      case mp.intro.intro
      R : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : Nontrivial R
      h : Not (IsField R)
      x : R
      nz : Ne x 0
      nu : Not (IsUnit x)
      ⊢ Exists fun I => And (LT.lt Bot.bot I) (LT.lt I Top.top)
    -/
    use Ideal.span {x}
    /-
      case h
      R : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : Nontrivial R
      h : Not (IsField R)
      x : R
      nz : Ne x 0
      nu : Not (IsUnit x)
      ⊢ And (LT.lt Bot.bot (Ideal.span (Singleton.singleton x))) (LT.lt (Ideal.span  …
    -/
    rw [bot_lt_iff_ne_bot, lt_top_iff_ne_top]
    /-
      case h
      R : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : Nontrivial R
      h : Not (IsField R)
      x : R
      nz : Ne x 0
      nu : Not (IsUnit x)
      ⊢ And (Ne (Ideal.span (Singleton.singleton x)) Bot.bot) (Ne (Ideal.span (Singl …
    -/
    exact ⟨mt Ideal.span_singleton_eq_bot.mp nz, mt Ideal.span_singleton_eq_top.mp nu⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : Nontrivial R
      ⊢ (Exists fun I => And (LT.lt Bot.bot I) (LT.lt I Top.top)) → Not (IsField R)
    -/
  · rintro ⟨I, bot_lt, lt_top⟩ hf
    /-
      case mpr.intro.intro
      R : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : Nontrivial R
      I : Ideal R
      bot_lt : LT.lt Bot.bot I
      lt_top : LT.lt I Top.top
      hf : IsField R
      ⊢ False
    -/
    obtain ⟨x, mem, ne_zero⟩ := SetLike.exists_of_lt bot_lt
    /-
      case mpr.intro.intro.intro.intro
      R : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : Nontrivial R
      I : Ideal R
      bot_lt : LT.lt Bot.bot I
      lt_top : LT.lt I Top.top
      hf : IsField R
      x : R
      mem : Membership.mem I x
      ne_zero : Not (Membership.mem Bot.bot x)
      ⊢ False
    -/
    rw [Submodule.mem_bot] at ne_zero
    /-
      case mpr.intro.intro.intro.intro
      R : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : Nontrivial R
      I : Ideal R
      bot_lt : LT.lt Bot.bot I
      lt_top : LT.lt I Top.top
      hf : IsField R
      x : R
      mem : Membership.mem I x
      ne_zero : Not (Eq x 0)
      ⊢ False
    -/
    obtain ⟨y, hy⟩ := hf.mul_inv_cancel ne_zero
    /-
      case mpr.intro.intro.intro.intro.intro
      R : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : Nontrivial R
      I : Ideal R
      bot_lt : LT.lt Bot.bot I
      lt_top : LT.lt I Top.top
      hf : IsField R
      x : R
      mem : Membership.mem I x
      ne_zero : Not (Eq x 0)
      y : R
      hy : Eq (HMul.hMul x y) 1
      ⊢ False
    -/
    rw [lt_top_iff_ne_top, Ne, Ideal.eq_top_iff_one, ← hy] at lt_top
    /-
      case mpr.intro.intro.intro.intro.intro
      R : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : Nontrivial R
      I : Ideal R
      bot_lt : LT.lt Bot.bot I
      hf : IsField R
      x : R
      mem : Membership.mem I x
      ne_zero : Not (Eq x 0)
      y : R
      lt_top : Not (Membership.mem I (HMul.hMul x y))
      hy : Eq (HMul.hMul x y) 1
      ⊢ False
    -/
    exact lt_top (I.mul_mem_right _ mem)
    /-
      🎉 no goals
    -/


theorem not_isField_iff_exists_prime [Nontrivial R] :
    ¬IsField R ↔ ∃ p : Ideal R, p ≠ ⊥ ∧ p.IsPrime :=
  not_isField_iff_exists_ideal_bot_lt_and_lt_top.trans
    ⟨fun ⟨I, bot_lt, lt_top⟩ =>
      let ⟨p, hp, le_p⟩ := I.exists_le_maximal (lt_top_iff_ne_top.mp lt_top)
      ⟨p, bot_lt_iff_ne_bot.mp (lt_of_lt_of_le bot_lt le_p), hp.isPrime⟩,
      fun ⟨p, ne_bot, Prime⟩ => ⟨p, bot_lt_iff_ne_bot.mpr ne_bot, lt_top_iff_ne_top.mpr Prime.1⟩⟩


/-- Also see `Ideal.isSimpleOrder` for the forward direction as an instance when `R` is a
division (semi)ring.

This result actually holds for all division semirings, but we lack the predicate to state it. -/
theorem isField_iff_isSimpleOrder_ideal : IsField R ↔ IsSimpleOrder (Ideal R) := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    ⊢ Iff (IsField R) (IsSimpleOrder (Ideal R))
  -/
  cases subsingleton_or_nontrivial R
  · exact
      ⟨fun h => (not_isField_of_subsingleton _ h).elim, fun h =>
        (false_of_nontrivial_of_subsingleton <| Ideal R).elim⟩
  /-
    case inr
    R : Type u_1
    inst✝ : CommSemiring R
    h✝ : Nontrivial R
    ⊢ Iff (IsField R) (IsSimpleOrder (Ideal R))
  -/
  rw [← not_iff_not, Ring.not_isField_iff_exists_ideal_bot_lt_and_lt_top, ← not_iff_not]
  /-
    case inr
    R : Type u_1
    inst✝ : CommSemiring R
    h✝ : Nontrivial R
    ⊢ Iff (Not (Exists fun I => And (LT.lt Bot.bot I) (LT.lt I Top.top))) (Not (No …
  -/
  push_neg
  /-
    case inr
    R : Type u_1
    inst✝ : CommSemiring R
    h✝ : Nontrivial R
    ⊢ Iff (∀ (I : Ideal R), LT.lt Bot.bot I → Not (LT.lt I Top.top)) (IsSimpleOrde …
  -/
  simp_rw [lt_top_iff_ne_top, bot_lt_iff_ne_bot, ← or_iff_not_imp_left, not_ne_iff]
  /-
    case inr
    R : Type u_1
    inst✝ : CommSemiring R
    h✝ : Nontrivial R
    ⊢ Iff (∀ (I : Ideal R), Or (Eq I Bot.bot) (Eq I Top.top)) (IsSimpleOrder (Idea …
  -/
  exact ⟨fun h => ⟨h⟩, fun h => h.2⟩
  /-
    🎉 no goals
  -/


/-- When a ring is not a field, the maximal ideals are nontrivial. -/
theorem ne_bot_of_isMaximal_of_not_isField [Nontrivial R] {M : Ideal R} (max : M.IsMaximal)
    (not_field : ¬IsField R) : M ≠ ⊥ := by
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    M : Ideal R
    max : M.IsMaximal
    not_field : Not (IsField R)
    ⊢ Ne M Bot.bot
  -/
  rintro h
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    M : Ideal R
    max : M.IsMaximal
    not_field : Not (IsField R)
    h : Eq M Bot.bot
    ⊢ False
  -/
  rw [h] at max
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    M : Ideal R
    max : Bot.bot.IsMaximal
    not_field : Not (IsField R)
    h : Eq M Bot.bot
    ⊢ False
  -/
  rcases max with ⟨⟨_h1, h2⟩⟩
  /-
    case mk.intro
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    M : Ideal R
    not_field : Not (IsField R)
    h : Eq M Bot.bot
    _h1 : Ne Bot.bot Top.top
    h2 : ∀ (b : Ideal R), LT.lt Bot.bot b → Eq b Top.top
    ⊢ False
  -/
  obtain ⟨I, hIbot, hItop⟩ := not_isField_iff_exists_ideal_bot_lt_and_lt_top.mp not_field
  /-
    case mk.intro.intro.intro
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    M : Ideal R
    not_field : Not (IsField R)
    h : Eq M Bot.bot
    _h1 : Ne Bot.bot Top.top
    h2 : ∀ (b : Ideal R), LT.lt Bot.bot b → Eq b Top.top
    I : Ideal R
    hIbot : LT.lt Bot.bot I
    hItop : LT.lt I Top.top
    ⊢ False
  -/
  exact ne_of_lt hItop (h2 I hIbot)
  /-
    🎉 no goals
  -/


theorem bot_lt_of_maximal (M : Ideal R) [hm : M.IsMaximal] (non_field : ¬IsField R) : ⊥ < M := by
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    M : Ideal R
    hm : M.IsMaximal
    non_field : Not (IsField R)
    ⊢ LT.lt Bot.bot M
  -/
  rcases Ring.not_isField_iff_exists_ideal_bot_lt_and_lt_top.1 non_field with ⟨I, Ibot, Itop⟩
  /-
    case intro.intro
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    M : Ideal R
    hm : M.IsMaximal
    non_field : Not (IsField R)
    I : Ideal R
    Ibot : LT.lt Bot.bot I
    Itop : LT.lt I Top.top
    ⊢ LT.lt Bot.bot M
  -/
  constructor; · simp
                 /-
                   🎉 no goals
                 -/
  /-
    case intro.intro.right
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    M : Ideal R
    hm : M.IsMaximal
    non_field : Not (IsField R)
    I : Ideal R
    Ibot : LT.lt Bot.bot I
    Itop : LT.lt I Top.top
    ⊢ Not (HasSubset.Subset ↑M ↑Bot.bot)
  -/
  intro mle
  /-
    case intro.intro.right
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    M : Ideal R
    hm : M.IsMaximal
    non_field : Not (IsField R)
    I : Ideal R
    Ibot : LT.lt Bot.bot I
    Itop : LT.lt I Top.top
    mle : HasSubset.Subset ↑M ↑Bot.bot
    ⊢ False
  -/
  apply lt_irrefl (⊤ : Ideal R)
  /-
    case intro.intro.right
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    M : Ideal R
    hm : M.IsMaximal
    non_field : Not (IsField R)
    I : Ideal R
    Ibot : LT.lt Bot.bot I
    Itop : LT.lt I Top.top
    mle : HasSubset.Subset ↑M ↑Bot.bot
    ⊢ LT.lt Top.top Top.top
  -/
  have : M = ⊥ := eq_bot_iff.mpr mle
  /-
    case intro.intro.right
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    M : Ideal R
    hm : M.IsMaximal
    non_field : Not (IsField R)
    I : Ideal R
    Ibot : LT.lt Bot.bot I
    Itop : LT.lt I Top.top
    mle : HasSubset.Subset ↑M ↑Bot.bot
    this : Eq M Bot.bot
    ⊢ LT.lt Top.top Top.top
  -/
  rw [← this] at Ibot
  /-
    case intro.intro.right
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    M : Ideal R
    hm : M.IsMaximal
    non_field : Not (IsField R)
    I : Ideal R
    Ibot : LT.lt M I
    Itop : LT.lt I Top.top
    mle : HasSubset.Subset ↑M ↑Bot.bot
    this : Eq M Bot.bot
    ⊢ LT.lt Top.top Top.top
  -/
  rwa [hm.1.2 I Ibot] at Itop
  /-
    🎉 no goals
  -/


