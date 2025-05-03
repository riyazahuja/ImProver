/--
Proposition I.2.3 + I.2.6 of [iversen]
A finite-type `R`-algebra `S` is (formally) unramified iff there exists a `t : S ⊗[R] S` satisfying
1. `t` annihilates every `1 ⊗ s - s ⊗ 1`.
2. the image of `t` is `1` under the map `S ⊗[R] S → S`.
-/
theorem iff_exists_tensorProduct [EssFiniteType R S] :
    FormallyUnramified R S ↔ ∃ t : S ⊗[R] S,
      (∀ s, ((1 : S) ⊗ₜ[R] s - s ⊗ₜ[R] (1 : S)) * t = 0) ∧ TensorProduct.lmul' R t = 1 := by
  rw [formallyUnramified_iff, KaehlerDifferential,
    Ideal.cotangent_subsingleton_iff, Ideal.isIdempotentElem_iff_of_fg _
      (KaehlerDifferential.ideal_fg R S)]
  have : ∀ t : S ⊗[R] S, TensorProduct.lmul' R t = 1 ↔ 1 - t ∈ KaehlerDifferential.ideal R S := by
    intro t
    simp only [KaehlerDifferential.ideal, RingHom.mem_ker, map_sub, map_one,
      sub_eq_zero, @eq_comm S 1]
  /-
    R : Type u_2
    S : Type u_3
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : Algebra.EssFiniteType R S
    this : ∀ (t : TensorProduct R S S), Iff (Eq ((Algebra.TensorProduct.lmul' R) t …
    ⊢ Iff (Exists fun e => And (IsIdempotentElem e) (Eq (KaehlerDifferential.ideal …
  -/
  simp_rw [this, ← KaehlerDifferential.span_range_eq_ideal]
  /-
    R : Type u_2
    S : Type u_3
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : Algebra.EssFiniteType R S
    this : ∀ (t : TensorProduct R S S), Iff (Eq ((Algebra.TensorProduct.lmul' R) t …
    ⊢ Iff (Exists fun e => And (IsIdempotentElem e) (Eq (Ideal.span (Set.range fun …
  -/
  constructor
    /-
      case mp
      R : Type u_2
      S : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      inst✝ : Algebra.EssFiniteType R S
      this : ∀ (t : TensorProduct R S S), Iff (Eq ((Algebra.TensorProduct.lmul' R) t …
      ⊢ (Exists fun e => And (IsIdempotentElem e) (Eq (Ideal.span (Set.range fun s = …
    -/
  · rintro ⟨e, he₁, he₂ : _ = Ideal.span _⟩
    /-
      case mp.intro.intro
      R : Type u_2
      S : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      inst✝ : Algebra.EssFiniteType R S
      this : ∀ (t : TensorProduct R S S), Iff (Eq ((Algebra.TensorProduct.lmul' R) t …
      e : TensorProduct R S S
      he₁ : IsIdempotentElem e
      he₂ : Eq (Ideal.span (Set.range fun s => HSub.hSub (TensorProduct.tmul R 1 s)  …
      ⊢ Exists fun t => And (∀ (s : S), Eq (HMul.hMul (HSub.hSub (TensorProduct.tmul …
    -/
    refine ⟨1 - e, ?_, ?_⟩
      /-
        case mp.intro.intro.refine_1
        R : Type u_2
        S : Type u_3
        inst✝³ : CommRing R
        inst✝² : CommRing S
        inst✝¹ : Algebra R S
        inst✝ : Algebra.EssFiniteType R S
        this : ∀ (t : TensorProduct R S S), Iff (Eq ((Algebra.TensorProduct.lmul' R) t …
        e : TensorProduct R S S
        he₁ : IsIdempotentElem e
        he₂ : Eq (Ideal.span (Set.range fun s => HSub.hSub (TensorProduct.tmul R 1 s)  …
        ⊢ ∀ (s : S), Eq (HMul.hMul (HSub.hSub (TensorProduct.tmul R 1 s) (TensorProduc …
      -/
    · intro s
      obtain ⟨x, hx⟩ : e ∣ 1 ⊗ₜ[R] s - s ⊗ₜ[R] 1 := by
        rw [← Ideal.mem_span_singleton, ← he₂]
        exact Ideal.subset_span ⟨s, rfl⟩
      /-
        case mp.intro.intro.refine_1.intro
        R : Type u_2
        S : Type u_3
        inst✝³ : CommRing R
        inst✝² : CommRing S
        inst✝¹ : Algebra R S
        inst✝ : Algebra.EssFiniteType R S
        this : ∀ (t : TensorProduct R S S), Iff (Eq ((Algebra.TensorProduct.lmul' R) t …
        e : TensorProduct R S S
        he₁ : IsIdempotentElem e
        he₂ : Eq (Ideal.span (Set.range fun s => HSub.hSub (TensorProduct.tmul R 1 s)  …
        s : S
        x : TensorProduct R S S
        hx : Eq (HSub.hSub (TensorProduct.tmul R 1 s) (TensorProduct.tmul R s 1)) (HMu …
        ⊢ Eq (HMul.hMul (HSub.hSub (TensorProduct.tmul R 1 s) (TensorProduct.tmul R s  …
      -/
      rw [hx, mul_comm, ← mul_assoc, sub_mul, one_mul, he₁.eq, sub_self, zero_mul]
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.intro.refine_2
        R : Type u_2
        S : Type u_3
        inst✝³ : CommRing R
        inst✝² : CommRing S
        inst✝¹ : Algebra R S
        inst✝ : Algebra.EssFiniteType R S
        this : ∀ (t : TensorProduct R S S), Iff (Eq ((Algebra.TensorProduct.lmul' R) t …
        e : TensorProduct R S S
        he₁ : IsIdempotentElem e
        he₂ : Eq (Ideal.span (Set.range fun s => HSub.hSub (TensorProduct.tmul R 1 s)  …
        ⊢ Membership.mem (Ideal.span (Set.range fun s => HSub.hSub (TensorProduct.tmul …
      -/
    · rw [sub_sub_cancel, he₂, Ideal.mem_span_singleton]
      /-
        🎉 no goals
      -/
    /-
      case mpr
      R : Type u_2
      S : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      inst✝ : Algebra.EssFiniteType R S
      this : ∀ (t : TensorProduct R S S), Iff (Eq ((Algebra.TensorProduct.lmul' R) t …
      ⊢ (Exists fun t => And (∀ (s : S), Eq (HMul.hMul (HSub.hSub (TensorProduct.tmu …
    -/
  · rintro ⟨t, ht₁, ht₂⟩
    /-
      case mpr.intro.intro
      R : Type u_2
      S : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      inst✝ : Algebra.EssFiniteType R S
      this : ∀ (t : TensorProduct R S S), Iff (Eq ((Algebra.TensorProduct.lmul' R) t …
      t : TensorProduct R S S
      ht₁ : ∀ (s : S), Eq (HMul.hMul (HSub.hSub (TensorProduct.tmul R 1 s) (TensorPr …
      ht₂ : Membership.mem (Ideal.span (Set.range fun s => HSub.hSub (TensorProduct. …
      ⊢ Exists fun e => And (IsIdempotentElem e) (Eq (Ideal.span (Set.range fun s => …
    -/
    use 1 - t
    /-
      case h
      R : Type u_2
      S : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      inst✝ : Algebra.EssFiniteType R S
      this : ∀ (t : TensorProduct R S S), Iff (Eq ((Algebra.TensorProduct.lmul' R) t …
      t : TensorProduct R S S
      ht₁ : ∀ (s : S), Eq (HMul.hMul (HSub.hSub (TensorProduct.tmul R 1 s) (TensorPr …
      ht₂ : Membership.mem (Ideal.span (Set.range fun s => HSub.hSub (TensorProduct. …
      ⊢ And (IsIdempotentElem (HSub.hSub 1 t)) (Eq (Ideal.span (Set.range fun s => H …
    -/
    rw [← sub_sub_self 1 t] at ht₁; generalize 1 - t = e at *
    /-
      case h
      R : Type u_2
      S : Type u_3
      inst✝³ : CommRing R
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      inst✝ : Algebra.EssFiniteType R S
      this : ∀ (t : TensorProduct R S S), Iff (Eq ((Algebra.TensorProduct.lmul' R) t …
      t e : TensorProduct R S S
      ht₁ : ∀ (s : S), Eq (HMul.hMul (HSub.hSub (TensorProduct.tmul R 1 s) (TensorPr …
      ht₂ : Membership.mem (Ideal.span (Set.range fun s => HSub.hSub (TensorProduct. …
      ⊢ And (IsIdempotentElem e) (Eq (Ideal.span (Set.range fun s => HSub.hSub (Tens …
    -/
    constructor
    · suffices e ∈ (Submodule.span (S ⊗[R] S) {1 - e}).annihilator by
        simpa [IsIdempotentElem, mul_sub, sub_eq_zero, eq_comm, -Ideal.submodule_span_eq,
          Submodule.mem_annihilator_span_singleton] using this
      exact (show Ideal.span _ ≤ _ by simpa only [Ideal.span_le, Set.range_subset_iff,
        Submodule.mem_annihilator_span_singleton, SetLike.mem_coe]) ht₂
      /-
        case h.right
        R : Type u_2
        S : Type u_3
        inst✝³ : CommRing R
        inst✝² : CommRing S
        inst✝¹ : Algebra R S
        inst✝ : Algebra.EssFiniteType R S
        this : ∀ (t : TensorProduct R S S), Iff (Eq ((Algebra.TensorProduct.lmul' R) t …
        t e : TensorProduct R S S
        ht₁ : ∀ (s : S), Eq (HMul.hMul (HSub.hSub (TensorProduct.tmul R 1 s) (TensorPr …
        ht₂ : Membership.mem (Ideal.span (Set.range fun s => HSub.hSub (TensorProduct. …
        ⊢ Eq (Ideal.span (Set.range fun s => HSub.hSub (TensorProduct.tmul R 1 s) (Ten …
      -/
    · apply le_antisymm <;> simp only [Ideal.submodule_span_eq, Ideal.mem_span_singleton, ht₂,
        Ideal.span_le, Set.singleton_subset_iff, SetLike.mem_coe, Set.range_subset_iff]
      /-
        case h.right.a
        R : Type u_2
        S : Type u_3
        inst✝³ : CommRing R
        inst✝² : CommRing S
        inst✝¹ : Algebra R S
        inst✝ : Algebra.EssFiniteType R S
        this : ∀ (t : TensorProduct R S S), Iff (Eq ((Algebra.TensorProduct.lmul' R) t …
        t e : TensorProduct R S S
        ht₁ : ∀ (s : S), Eq (HMul.hMul (HSub.hSub (TensorProduct.tmul R 1 s) (TensorPr …
        ht₂ : Membership.mem (Ideal.span (Set.range fun s => HSub.hSub (TensorProduct. …
        ⊢ ∀ (y : S), Dvd.dvd e (HSub.hSub (TensorProduct.tmul R 1 y) (TensorProduct.tm …
      -/
      intro s
      /-
        case h.right.a
        R : Type u_2
        S : Type u_3
        inst✝³ : CommRing R
        inst✝² : CommRing S
        inst✝¹ : Algebra R S
        inst✝ : Algebra.EssFiniteType R S
        this : ∀ (t : TensorProduct R S S), Iff (Eq ((Algebra.TensorProduct.lmul' R) t …
        t e : TensorProduct R S S
        ht₁ : ∀ (s : S), Eq (HMul.hMul (HSub.hSub (TensorProduct.tmul R 1 s) (TensorPr …
        ht₂ : Membership.mem (Ideal.span (Set.range fun s => HSub.hSub (TensorProduct. …
        s : S
        ⊢ Dvd.dvd e (HSub.hSub (TensorProduct.tmul R 1 s) (TensorProduct.tmul R s 1))
      -/
      use 1 ⊗ₜ[R] s - s ⊗ₜ[R] 1
      /-
        case h
        R : Type u_2
        S : Type u_3
        inst✝³ : CommRing R
        inst✝² : CommRing S
        inst✝¹ : Algebra R S
        inst✝ : Algebra.EssFiniteType R S
        this : ∀ (t : TensorProduct R S S), Iff (Eq ((Algebra.TensorProduct.lmul' R) t …
        t e : TensorProduct R S S
        ht₁ : ∀ (s : S), Eq (HMul.hMul (HSub.hSub (TensorProduct.tmul R 1 s) (TensorPr …
        ht₂ : Membership.mem (Ideal.span (Set.range fun s => HSub.hSub (TensorProduct. …
        s : S
        ⊢ Eq (HSub.hSub (TensorProduct.tmul R 1 s) (TensorProduct.tmul R s 1)) (HMul.h …
      -/
      linear_combination ht₁ s
      /-
        🎉 no goals
      -/


lemma finite_of_free_aux (I) [DecidableEq I] (b : Basis I R S)
    (f : I →₀ S) (x : S) (a : I → I →₀ R) (ha : a = fun i ↦ b.repr (b i * x)) :
    (1 ⊗ₜ[R] x * Finsupp.sum f fun i y ↦ y ⊗ₜ[R] b i) =
      Finset.sum (f.support.biUnion fun i ↦ (a i).support) fun k ↦
    Finsupp.sum (b.repr (f.sum fun i y ↦ a i k • y)) fun j c ↦ c • b j ⊗ₜ[R] b k := by
  /-
    R : Type u_3
    S : Type u_4
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    I : Type u_2
    inst✝ : DecidableEq I
    b : Basis I R S
    f : Finsupp I S
    x : S
    a : I → Finsupp I R
    ha : Eq a fun i => b.repr (HMul.hMul (b i) x)
    ⊢ Eq (HMul.hMul (TensorProduct.tmul R 1 x) (f.sum fun i y => TensorProduct.tmu …
  -/
  rw [Finsupp.sum, Finset.mul_sum]
  /-
    R : Type u_3
    S : Type u_4
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    I : Type u_2
    inst✝ : DecidableEq I
    b : Basis I R S
    f : Finsupp I S
    x : S
    a : I → Finsupp I R
    ha : Eq a fun i => b.repr (HMul.hMul (b i) x)
    ⊢ Eq (f.support.sum fun i => HMul.hMul (TensorProduct.tmul R 1 x) (TensorProdu …
  -/
  subst ha
  /-
    R : Type u_3
    S : Type u_4
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    I : Type u_2
    inst✝ : DecidableEq I
    b : Basis I R S
    f : Finsupp I S
    x : S
    ⊢ Eq (f.support.sum fun i => HMul.hMul (TensorProduct.tmul R 1 x) (TensorProdu …
  -/
  let a i := b.repr (b i * x)
  conv_lhs =>
    simp only [TensorProduct.tmul_mul_tmul, one_mul, mul_comm x (b _),
      ← show ∀ i, Finsupp.linearCombination _ b (a i) = b i * x from
          fun _ ↦ b.linearCombination_repr _]
  conv_lhs => simp only [Finsupp.linearCombination, Finsupp.coe_lsum,
    LinearMap.coe_smulRight, LinearMap.id_coe, id_eq, Finsupp.sum, TensorProduct.tmul_sum,
    ← TensorProduct.smul_tmul]
  have h₁ : ∀ k,
    (Finsupp.sum (Finsupp.sum f fun i y ↦ a i k • b.repr y) fun j z ↦ z • b j ⊗ₜ[R] b k) =
      (f.sum fun i y ↦ (b.repr y).sum fun j z ↦ a i k • z • b j ⊗ₜ[R] b k) := by
    intro i
    rw [Finsupp.sum_sum_index]
    congr
    ext j s
    rw [Finsupp.sum_smul_index]
    simp only [mul_smul, Finsupp.sum, ← Finset.smul_sum]
    · intro; simp only [zero_smul]
    · intro; simp only [zero_smul]
    · intros; simp only [add_smul]
  have h₂ : ∀ (x : S), ((b.repr x).support.sum fun a ↦ b.repr x a • b a) = x := by
    simpa only [Finsupp.linearCombination_apply, Finsupp.sum] using b.linearCombination_repr
  /-
    R : Type u_3
    S : Type u_4
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    I : Type u_2
    inst✝ : DecidableEq I
    b : Basis I R S
    f : Finsupp I S
    x : S
    a : I → Finsupp I R := fun i => b.repr (HMul.hMul (b i) x)
    h₁ : ∀ (k : I), Eq ((f.sum fun i y => HSMul.hSMul ((a i) k) (b.repr y)).sum fu …
    h₂ : ∀ (x : S), Eq ((b.repr x).support.sum fun a => HSMul.hSMul ((b.repr x) a) …
    ⊢ Eq (f.support.sum fun x => (a x).support.sum fun x_1 => TensorProduct.tmul R …
  -/
  simp only [a] at h₁
  simp_rw [map_finsupp_sum, map_smul, h₁, Finsupp.sum, Finset.sum_comm (t := f.support),
    TensorProduct.smul_tmul', ← TensorProduct.sum_tmul, ← Finset.smul_sum, h₂]
  /-
    R : Type u_3
    S : Type u_4
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    I : Type u_2
    inst✝ : DecidableEq I
    b : Basis I R S
    f : Finsupp I S
    x : S
    a : I → Finsupp I R := fun i => b.repr (HMul.hMul (b i) x)
    h₁ : ∀ (k : I), Eq ((f.sum fun i y => HSMul.hSMul ((b.repr (HMul.hMul (b i) x) …
    h₂ : ∀ (x : S), Eq ((b.repr x).support.sum fun a => HSMul.hSMul ((b.repr x) a) …
    ⊢ Eq (f.support.sum fun x => (a x).support.sum fun x_1 => TensorProduct.tmul R …
  -/
  apply Finset.sum_congr rfl
  /-
    R : Type u_3
    S : Type u_4
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    I : Type u_2
    inst✝ : DecidableEq I
    b : Basis I R S
    f : Finsupp I S
    x : S
    a : I → Finsupp I R := fun i => b.repr (HMul.hMul (b i) x)
    h₁ : ∀ (k : I), Eq ((f.sum fun i y => HSMul.hSMul ((b.repr (HMul.hMul (b i) x) …
    h₂ : ∀ (x : S), Eq ((b.repr x).support.sum fun a => HSMul.hSMul ((b.repr x) a) …
    ⊢ ∀ (x_1 : I), Membership.mem f.support x_1 → Eq ((a x_1).support.sum fun x => …
  -/
  intros i hi
  /-
    R : Type u_3
    S : Type u_4
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    I : Type u_2
    inst✝ : DecidableEq I
    b : Basis I R S
    f : Finsupp I S
    x : S
    a : I → Finsupp I R := fun i => b.repr (HMul.hMul (b i) x)
    h₁ : ∀ (k : I), Eq ((f.sum fun i y => HSMul.hSMul ((b.repr (HMul.hMul (b i) x) …
    h₂ : ∀ (x : S), Eq ((b.repr x).support.sum fun a => HSMul.hSMul ((b.repr x) a) …
    i : I
    hi : Membership.mem f.support i
    ⊢ Eq ((a i).support.sum fun x => TensorProduct.tmul R (HSMul.hSMul ((a i) x) ( …
  -/
  apply Finset.sum_subset_zero_on_sdiff
    /-
      case h
      R : Type u_3
      S : Type u_4
      inst✝³ : CommRing R
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      I : Type u_2
      inst✝ : DecidableEq I
      b : Basis I R S
      f : Finsupp I S
      x : S
      a : I → Finsupp I R := fun i => b.repr (HMul.hMul (b i) x)
      h₁ : ∀ (k : I), Eq ((f.sum fun i y => HSMul.hSMul ((b.repr (HMul.hMul (b i) x) …
      h₂ : ∀ (x : S), Eq ((b.repr x).support.sum fun a => HSMul.hSMul ((b.repr x) a) …
      i : I
      hi : Membership.mem f.support i
      ⊢ HasSubset.Subset (a i).support (f.support.biUnion fun i => (b.repr (HMul.hMu …
    -/
  · exact Finset.subset_biUnion_of_mem (fun i ↦ (a i).support) hi
    /-
      🎉 no goals
    -/
  · simp only [a, Finset.mem_sdiff, Finset.mem_biUnion, Finsupp.mem_support_iff, ne_eq, not_not,
      and_imp, forall_exists_index]
    /-
      case hg
      R : Type u_3
      S : Type u_4
      inst✝³ : CommRing R
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      I : Type u_2
      inst✝ : DecidableEq I
      b : Basis I R S
      f : Finsupp I S
      x : S
      a : I → Finsupp I R := fun i => b.repr (HMul.hMul (b i) x)
      h₁ : ∀ (k : I), Eq ((f.sum fun i y => HSMul.hSMul ((b.repr (HMul.hMul (b i) x) …
      h₂ : ∀ (x : S), Eq ((b.repr x).support.sum fun a => HSMul.hSMul ((b.repr x) a) …
      i : I
      hi : Membership.mem f.support i
      ⊢ ∀ (x_1 x_2 : I), Not (Eq (f x_2) 0) → Not (Eq ((b.repr (HMul.hMul (b x_2) x) …
    -/
    simp (config := {contextual := true})
    /-
      🎉 no goals
    -/
    /-
      case hfg
      R : Type u_3
      S : Type u_4
      inst✝³ : CommRing R
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      I : Type u_2
      inst✝ : DecidableEq I
      b : Basis I R S
      f : Finsupp I S
      x : S
      a : I → Finsupp I R := fun i => b.repr (HMul.hMul (b i) x)
      h₁ : ∀ (k : I), Eq ((f.sum fun i y => HSMul.hSMul ((b.repr (HMul.hMul (b i) x) …
      h₂ : ∀ (x : S), Eq ((b.repr x).support.sum fun a => HSMul.hSMul ((b.repr x) a) …
      i : I
      hi : Membership.mem f.support i
      ⊢ ∀ (x_1 : I), Membership.mem (a i).support x_1 → Eq (TensorProduct.tmul R (HS …
    -/
  · exact fun _ _ ↦ rfl
    /-
      🎉 no goals
    -/


variable (R S) in
/--
A finite-type `R`-algebra `S` is (formally) unramified iff there exists a `t : S ⊗[R] S` satisfying
1. `t` annihilates every `1 ⊗ s - s ⊗ 1`.
2. the image of `t` is `1` under the map `S ⊗[R] S → S`.
See `Algebra.FormallyUnramified.iff_exists_tensorProduct`.
This is the choice of such a `t`.
-/
noncomputable
def elem : S ⊗[R] S :=
  (iff_exists_tensorProduct.mp inferInstance).choose


lemma one_tmul_sub_tmul_one_mul_elem
    (s : S) : (1 ⊗ₜ s - s ⊗ₜ 1) * elem R S = 0 :=
  (iff_exists_tensorProduct.mp inferInstance).choose_spec.1 s


lemma one_tmul_mul_elem
    (s : S) : (1 ⊗ₜ s) * elem R S = (s ⊗ₜ 1) * elem R S := by
  /-
    R : Type u_3
    S : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : Algebra.FormallyUnramified R S
    inst✝ : Algebra.EssFiniteType R S
    s : S
    ⊢ Eq (HMul.hMul (TensorProduct.tmul R 1 s) (Algebra.FormallyUnramified.elem R  …
  -/
  rw [← sub_eq_zero, ← sub_mul, one_tmul_sub_tmul_one_mul_elem]
  /-
    🎉 no goals
  -/


lemma lmul_elem :
    TensorProduct.lmul' R (elem R S) = 1 :=
  (iff_exists_tensorProduct.mp inferInstance).choose_spec.2



/-- An unramified free algebra is finitely generated. Iversen I.2.8 -/
lemma finite_of_free [Module.Free R S] : Module.Finite R S := by
  classical
  let I := Module.Free.ChooseBasisIndex R S
  -- Let `bᵢ` be an `R`-basis of `S`.
  let b : Basis I R S := Module.Free.chooseBasis R S
  -- Let `∑ₛ fᵢ ⊗ bᵢ : S ⊗[R] S` (summing over some finite `s`) be an element such that
  -- `∑ₛ fᵢbᵢ = 1` and `∀ x : S, xfᵢ ⊗ bᵢ = aᵢ ⊗ xfᵢ` which exists since `S` is unramified over `R`.
  have ⟨f, hf⟩ : ∃ (a : I →₀ S), elem R S = a.sum (fun i x ↦ x ⊗ₜ b i) := by
    let b' := ((Basis.singleton PUnit.{1} S).tensorProduct b).reindex (Equiv.punitProd I)
    use b'.repr (elem R S)
    conv_lhs => rw [← b'.linearCombination_repr (elem R S), Finsupp.linearCombination_apply]
    congr! with _ i x
    simp [b', Basis.tensorProduct, TensorProduct.smul_tmul']
  constructor
  -- I claim that `{ fᵢbⱼ | i, j ∈ s }` spans `S` over `R`.
  use Finset.image₂ (fun i j ↦ f i * b j) f.support f.support
  rw [← top_le_iff]
  -- For all `x : S`, let `bᵢx = ∑ aᵢⱼbⱼ`.
  rintro x -
  let a : I → I →₀ R := fun i ↦ b.repr (b i * x)
  -- Consider `F` such that `fⱼx = ∑ Fᵢⱼbⱼ`.
  let F : I →₀ I →₀ R := Finsupp.onFinset f.support (fun j ↦ b.repr (x * f j))
    (fun j ↦ not_imp_comm.mp fun hj ↦ by simp [Finsupp.not_mem_support_iff.mp hj])
  have hG : ∀ j ∉ (Finset.biUnion f.support fun i ↦ (a i).support),
      b.repr (f.sum (fun i y ↦ a i j • y)) = 0 := by
    intros j hj
    simp only [Finset.mem_biUnion, Finsupp.mem_support_iff, ne_eq, not_exists, not_and,
      not_not] at hj
    simp only [Finsupp.sum]
    trans b.repr (f.support.sum (fun _ ↦ 0))
    · refine congr_arg b.repr (Finset.sum_congr rfl ?_)
      simp only [Finsupp.mem_support_iff]
      intro i hi
      rw [hj i hi, zero_smul]
    · simp only [Finset.sum_const_zero, map_zero]
  -- And `G` such that `∑ₛ aᵢⱼfᵢ = ∑ Gᵢⱼbⱼ`, where `aᵢⱼ` are the coefficients `bᵢx = ∑ aᵢⱼbⱼ`.
  let G : I →₀ I →₀ R := Finsupp.onFinset (Finset.biUnion f.support (fun i ↦ (a i).support))
    (fun j ↦ b.repr (f.sum (fun i y ↦ a i j • y)))
    (fun j ↦ not_imp_comm.mp (hG j))
  -- Then `∑ Fᵢⱼ(bⱼ ⊗ bᵢ) = ∑ fⱼx ⊗ bᵢ = ∑ fⱼ ⊗ xbᵢ = ∑ aᵢⱼ(fⱼ ⊗ bᵢ) = ∑ Gᵢⱼ(bⱼ ⊗ bᵢ)`.
  -- Since `bⱼ ⊗ bᵢ` forms an `R`-basis of `S ⊗ S`, we conclude that `F = G`.
  have : F = G := by
    apply Finsupp.finsuppProdEquiv.symm.injective
    apply (Finsupp.equivCongrLeft (Equiv.prodComm I I)).injective
    apply (b.tensorProduct b).repr.symm.injective
    simp only [Basis.repr_symm_apply, Finsupp.coe_lsum, LinearMap.coe_smulRight,
      LinearMap.id_coe, id_eq, Basis.tensorProduct_apply, Finsupp.finsuppProdEquiv,
      Equiv.coe_fn_symm_mk, Finsupp.uncurry, map_finsupp_sum,
      Finsupp.linearCombination_single, Basis.tensorProduct_apply, Finsupp.equivCongrLeft_apply,
      Finsupp.linearCombination_equivMapDomain, Equiv.coe_prodComm]
    rw [Finsupp.onFinset_sum, Finsupp.onFinset_sum]
    simp only [Function.comp_apply, Prod.swap_prod_mk, Basis.tensorProduct_apply]
    have : ∀ i, ((b.repr (x * f i)).sum fun j k ↦ k • b j ⊗ₜ[R] b i) = (x * f i) ⊗ₜ[R] b i := by
      intro i
      simp_rw [Finsupp.sum, TensorProduct.smul_tmul', ← TensorProduct.sum_tmul]
      congr 1
      exact b.linearCombination_repr _
    trans (x ⊗ₜ 1) * elem R S
    · simp_rw [this, hf, Finsupp.sum, Finset.mul_sum, TensorProduct.tmul_mul_tmul, one_mul]
    · rw [← one_tmul_mul_elem, hf, finite_of_free_aux]
      rfl
    · intro; simp
    · intro; simp
  -- In particular, `fⱼx = ∑ Fᵢⱼbⱼ = ∑ Gᵢⱼbⱼ = ∑ₛ aᵢⱼfᵢ` for all `j`.
  have : ∀ j, x * f j = f.sum fun i y ↦ a i j • y := by
    intro j
    apply b.repr.injective
    exact DFunLike.congr_fun this j
  -- Since `∑ₛ fⱼbⱼ = 1`, `x = ∑ₛ aᵢⱼfᵢbⱼ` is indeed in the span of `{ fᵢbⱼ | i, j ∈ s }`.
  rw [← mul_one x, ← @lmul_elem R, hf, map_finsupp_sum, Finsupp.sum, Finset.mul_sum]
  simp only [TensorProduct.lmul'_apply_tmul, Finset.coe_image₂, ← mul_assoc, this,
    Finsupp.sum, Finset.sum_mul, smul_mul_assoc]
  apply Submodule.sum_mem; intro i hi
  apply Submodule.sum_mem; intro j hj
  apply Submodule.smul_mem
  apply Submodule.subset_span
  use j, hj, i, hi


/--
Proposition I.2.3 of [iversen]
If `S` is an unramified `R`-algebra, and `M` is a `S`-module, then the map
`S ⊗[R] M →ₗ[S] M` taking `(b, m) ↦ b • m` admits a `S`-linear section. -/
noncomputable
def sec :
    M →ₗ[S] S ⊗[R] M where
  __ := ((TensorProduct.AlgebraTensorModule.mapBilinear R S S S S S M
    LinearMap.id).flip (elem R S)).comp (lsmul R R M).toLinearMap.flip
  map_smul' r m := by
    simp only [AddHom.toFun_eq_coe, LinearMap.coe_toAddHom, LinearMap.coe_comp, Function.comp_apply,
      LinearMap.flip_apply, TensorProduct.AlgebraTensorModule.mapBilinear_apply, RingHom.id_apply]
    trans (TensorProduct.AlgebraTensorModule.map (LinearMap.id (R := S) (M := S))
      ((LinearMap.flip (AlgHom.toLinearMap (lsmul R R M))) m)) ((1 ⊗ₜ r) * elem R S)
      /-
        R : Type ?u.190352
        S : Type ?u.190355
        inst✝⁸ : CommRing R
        inst✝⁷ : CommRing S
        inst✝⁶ : Algebra R S
        M : Type u_1
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : Module R M
        inst✝³ : Module S M
        inst✝² : IsScalarTower R S M
        inst✝¹ : Algebra.FormallyUnramified R S
        inst✝ : Algebra.EssFiniteType R S
        r : S
        m : M
        ⊢ Eq ((TensorProduct.AlgebraTensorModule.map LinearMap.id ((Algebra.lsmul R R  …
      -/
    · induction' elem R S using TensorProduct.induction_on
        /-
          case zero
          R : Type ?u.190352
          S : Type ?u.190355
          inst✝⁸ : CommRing R
          inst✝⁷ : CommRing S
          inst✝⁶ : Algebra R S
          M : Type u_1
          inst✝⁵ : AddCommGroup M
          inst✝⁴ : Module R M
          inst✝³ : Module S M
          inst✝² : IsScalarTower R S M
          inst✝¹ : Algebra.FormallyUnramified R S
          inst✝ : Algebra.EssFiniteType R S
          r : S
          m : M
          ⊢ Eq ((TensorProduct.AlgebraTensorModule.map LinearMap.id ((Algebra.lsmul R R  …
        -/
      · simp
        /-
          🎉 no goals
        -/
        /-
          case tmul
          R : Type ?u.190352
          S : Type ?u.190355
          inst✝⁸ : CommRing R
          inst✝⁷ : CommRing S
          inst✝⁶ : Algebra R S
          M : Type u_1
          inst✝⁵ : AddCommGroup M
          inst✝⁴ : Module R M
          inst✝³ : Module S M
          inst✝² : IsScalarTower R S M
          inst✝¹ : Algebra.FormallyUnramified R S
          inst✝ : Algebra.EssFiniteType R S
          r : S
          m : M
          x✝ y✝ : S
          ⊢ Eq ((TensorProduct.AlgebraTensorModule.map LinearMap.id ((Algebra.lsmul R R  …
        -/
      · simp [smul_comm r]
        /-
          🎉 no goals
        -/
        /-
          case add
          R : Type ?u.190352
          S : Type ?u.190355
          inst✝⁸ : CommRing R
          inst✝⁷ : CommRing S
          inst✝⁶ : Algebra R S
          M : Type u_1
          inst✝⁵ : AddCommGroup M
          inst✝⁴ : Module R M
          inst✝³ : Module S M
          inst✝² : IsScalarTower R S M
          inst✝¹ : Algebra.FormallyUnramified R S
          inst✝ : Algebra.EssFiniteType R S
          r : S
          m : M
          x✝ y✝ : TensorProduct R S S
          a✝¹ : Eq ((TensorProduct.AlgebraTensorModule.map LinearMap.id ((Algebra.lsmul  …
          a✝ : Eq ((TensorProduct.AlgebraTensorModule.map LinearMap.id ((Algebra.lsmul R …
          ⊢ Eq ((TensorProduct.AlgebraTensorModule.map LinearMap.id ((Algebra.lsmul R R  …
        -/
      · simp only [map_add, mul_add, *]
        /-
          🎉 no goals
        -/
      /-
        R : Type ?u.190352
        S : Type ?u.190355
        inst✝⁸ : CommRing R
        inst✝⁷ : CommRing S
        inst✝⁶ : Algebra R S
        M : Type u_1
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : Module R M
        inst✝³ : Module S M
        inst✝² : IsScalarTower R S M
        inst✝¹ : Algebra.FormallyUnramified R S
        inst✝ : Algebra.EssFiniteType R S
        r : S
        m : M
        ⊢ Eq ((TensorProduct.AlgebraTensorModule.map LinearMap.id ((Algebra.lsmul R R  …
      -/
    · have := one_tmul_sub_tmul_one_mul_elem (R := R) r
      /-
        R : Type ?u.190352
        S : Type ?u.190355
        inst✝⁸ : CommRing R
        inst✝⁷ : CommRing S
        inst✝⁶ : Algebra R S
        M : Type u_1
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : Module R M
        inst✝³ : Module S M
        inst✝² : IsScalarTower R S M
        inst✝¹ : Algebra.FormallyUnramified R S
        inst✝ : Algebra.EssFiniteType R S
        r : S
        m : M
        this : Eq (HMul.hMul (HSub.hSub (TensorProduct.tmul R 1 r) (TensorProduct.tmul …
        ⊢ Eq ((TensorProduct.AlgebraTensorModule.map LinearMap.id ((Algebra.lsmul R R  …
      -/
      rw [sub_mul, sub_eq_zero] at this
      /-
        R : Type ?u.190352
        S : Type ?u.190355
        inst✝⁸ : CommRing R
        inst✝⁷ : CommRing S
        inst✝⁶ : Algebra R S
        M : Type u_1
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : Module R M
        inst✝³ : Module S M
        inst✝² : IsScalarTower R S M
        inst✝¹ : Algebra.FormallyUnramified R S
        inst✝ : Algebra.EssFiniteType R S
        r : S
        m : M
        this : Eq (HMul.hMul (TensorProduct.tmul R 1 r) (Algebra.FormallyUnramified.el …
        ⊢ Eq ((TensorProduct.AlgebraTensorModule.map LinearMap.id ((Algebra.lsmul R R  …
      -/
      rw [this]
      /-
        R : Type ?u.190352
        S : Type ?u.190355
        inst✝⁸ : CommRing R
        inst✝⁷ : CommRing S
        inst✝⁶ : Algebra R S
        M : Type u_1
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : Module R M
        inst✝³ : Module S M
        inst✝² : IsScalarTower R S M
        inst✝¹ : Algebra.FormallyUnramified R S
        inst✝ : Algebra.EssFiniteType R S
        r : S
        m : M
        this : Eq (HMul.hMul (TensorProduct.tmul R 1 r) (Algebra.FormallyUnramified.el …
        ⊢ Eq ((TensorProduct.AlgebraTensorModule.map LinearMap.id ((Algebra.lsmul R R  …
      -/
      induction' elem R S using TensorProduct.induction_on
        /-
          case zero
          R : Type ?u.190352
          S : Type ?u.190355
          inst✝⁸ : CommRing R
          inst✝⁷ : CommRing S
          inst✝⁶ : Algebra R S
          M : Type u_1
          inst✝⁵ : AddCommGroup M
          inst✝⁴ : Module R M
          inst✝³ : Module S M
          inst✝² : IsScalarTower R S M
          inst✝¹ : Algebra.FormallyUnramified R S
          inst✝ : Algebra.EssFiniteType R S
          r : S
          m : M
          this : Eq (HMul.hMul (TensorProduct.tmul R 1 r) (Algebra.FormallyUnramified.el …
          ⊢ Eq ((TensorProduct.AlgebraTensorModule.map LinearMap.id ((Algebra.lsmul R R  …
        -/
      · simp
        /-
          🎉 no goals
        -/
        /-
          case tmul
          R : Type ?u.190352
          S : Type ?u.190355
          inst✝⁸ : CommRing R
          inst✝⁷ : CommRing S
          inst✝⁶ : Algebra R S
          M : Type u_1
          inst✝⁵ : AddCommGroup M
          inst✝⁴ : Module R M
          inst✝³ : Module S M
          inst✝² : IsScalarTower R S M
          inst✝¹ : Algebra.FormallyUnramified R S
          inst✝ : Algebra.EssFiniteType R S
          r : S
          m : M
          this : Eq (HMul.hMul (TensorProduct.tmul R 1 r) (Algebra.FormallyUnramified.el …
          x✝ y✝ : S
          ⊢ Eq ((TensorProduct.AlgebraTensorModule.map LinearMap.id ((Algebra.lsmul R R  …
        -/
      · simp [TensorProduct.smul_tmul']
        /-
          🎉 no goals
        -/
        /-
          case add
          R : Type ?u.190352
          S : Type ?u.190355
          inst✝⁸ : CommRing R
          inst✝⁷ : CommRing S
          inst✝⁶ : Algebra R S
          M : Type u_1
          inst✝⁵ : AddCommGroup M
          inst✝⁴ : Module R M
          inst✝³ : Module S M
          inst✝² : IsScalarTower R S M
          inst✝¹ : Algebra.FormallyUnramified R S
          inst✝ : Algebra.EssFiniteType R S
          r : S
          m : M
          this : Eq (HMul.hMul (TensorProduct.tmul R 1 r) (Algebra.FormallyUnramified.el …
          x✝ y✝ : TensorProduct R S S
          a✝¹ : Eq ((TensorProduct.AlgebraTensorModule.map LinearMap.id ((Algebra.lsmul  …
          a✝ : Eq ((TensorProduct.AlgebraTensorModule.map LinearMap.id ((Algebra.lsmul R …
          ⊢ Eq ((TensorProduct.AlgebraTensorModule.map LinearMap.id ((Algebra.lsmul R R  …
        -/
      · simp only [map_add, smul_add, mul_add, *]
        /-
          🎉 no goals
        -/


lemma comp_sec :
    (TensorProduct.AlgebraTensorModule.lift
      ((lsmul S S M).toLinearMap.flip.restrictScalars R).flip).comp (sec R S M) =
      LinearMap.id := by
  /-
    R : Type u_3
    S : Type u_2
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    M : Type u_1
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module S M
    inst✝² : IsScalarTower R S M
    inst✝¹ : Algebra.FormallyUnramified R S
    inst✝ : Algebra.EssFiniteType R S
    ⊢ Eq ((TensorProduct.AlgebraTensorModule.lift (↑R (Algebra.lsmul S S M).toLine …
  -/
  ext x
  simp only [sec, LinearMap.coe_comp, LinearMap.coe_mk, LinearMap.coe_toAddHom,
    Function.comp_apply, LinearMap.flip_apply, TensorProduct.AlgebraTensorModule.mapBilinear_apply,
    TensorProduct.AlgebraTensorModule.lift_apply, LinearMap.id_coe, id_eq]
  /-
    case h
    R : Type u_3
    S : Type u_2
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    M : Type u_1
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module S M
    inst✝² : IsScalarTower R S M
    inst✝¹ : Algebra.FormallyUnramified R S
    inst✝ : Algebra.EssFiniteType R S
    x : M
    ⊢ Eq ((_root_.TensorProduct.lift (↑R (↑R (Algebra.lsmul S S M).toLinearMap.fli …
  -/
  trans (TensorProduct.lmul' R (elem R S)) • x
    /-
      R : Type u_3
      S : Type u_2
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Algebra R S
      M : Type u_1
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : Module S M
      inst✝² : IsScalarTower R S M
      inst✝¹ : Algebra.FormallyUnramified R S
      inst✝ : Algebra.EssFiniteType R S
      x : M
      ⊢ Eq ((_root_.TensorProduct.lift (↑R (↑R (Algebra.lsmul S S M).toLinearMap.fli …
    -/
  · induction' elem R S using TensorProduct.induction_on with r s y z hy hz
      /-
        case zero
        R : Type u_3
        S : Type u_2
        inst✝⁸ : CommRing R
        inst✝⁷ : CommRing S
        inst✝⁶ : Algebra R S
        M : Type u_1
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : Module R M
        inst✝³ : Module S M
        inst✝² : IsScalarTower R S M
        inst✝¹ : Algebra.FormallyUnramified R S
        inst✝ : Algebra.EssFiniteType R S
        x : M
        ⊢ Eq ((_root_.TensorProduct.lift (↑R (↑R (Algebra.lsmul S S M).toLinearMap.fli …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case tmul
        R : Type u_3
        S : Type u_2
        inst✝⁸ : CommRing R
        inst✝⁷ : CommRing S
        inst✝⁶ : Algebra R S
        M : Type u_1
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : Module R M
        inst✝³ : Module S M
        inst✝² : IsScalarTower R S M
        inst✝¹ : Algebra.FormallyUnramified R S
        inst✝ : Algebra.EssFiniteType R S
        x : M
        r s : S
        ⊢ Eq ((_root_.TensorProduct.lift (↑R (↑R (Algebra.lsmul S S M).toLinearMap.fli …
      -/
    · simp [mul_smul, smul_comm r s]
      /-
        🎉 no goals
      -/
      /-
        case add
        R : Type u_3
        S : Type u_2
        inst✝⁸ : CommRing R
        inst✝⁷ : CommRing S
        inst✝⁶ : Algebra R S
        M : Type u_1
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : Module R M
        inst✝³ : Module S M
        inst✝² : IsScalarTower R S M
        inst✝¹ : Algebra.FormallyUnramified R S
        inst✝ : Algebra.EssFiniteType R S
        x : M
        y z : TensorProduct R S S
        hy : Eq ((_root_.TensorProduct.lift (↑R (↑R (Algebra.lsmul S S M).toLinearMap. …
        hz : Eq ((_root_.TensorProduct.lift (↑R (↑R (Algebra.lsmul S S M).toLinearMap. …
        ⊢ Eq ((_root_.TensorProduct.lift (↑R (↑R (Algebra.lsmul S S M).toLinearMap.fli …
      -/
    · simp [hy, hz, add_smul]
      /-
        🎉 no goals
      -/
    /-
      R : Type u_3
      S : Type u_2
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Algebra R S
      M : Type u_1
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : Module S M
      inst✝² : IsScalarTower R S M
      inst✝¹ : Algebra.FormallyUnramified R S
      inst✝ : Algebra.EssFiniteType R S
      x : M
      ⊢ Eq (HSMul.hSMul ((Algebra.TensorProduct.lmul' R) (Algebra.FormallyUnramified …
    -/
  · rw [lmul_elem, one_smul]
    /-
      🎉 no goals
    -/


/-- If `S` is an unramified `R`-algebra, then `R`-flat implies `S`-flat. Iversen I.2.7 -/
lemma flat_of_restrictScalars [Module.Flat R M] : Module.Flat S M :=
  Module.Flat.of_retract _ _ _ _ _ (comp_sec R S M)


/-- If `S` is an unramified `R`-algebra, then `R`-projective implies `S`-projective. -/
lemma projective_of_restrictScalars [Module.Projective R M] : Module.Projective S M :=
  Module.Projective.of_split _ _ (comp_sec R S M)


