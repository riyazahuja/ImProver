instance instHasAntidiagonal : Finset.HasAntidiagonal (Additive ℕ+) :=
  /- The set of divisors of a positive natural number.
This is `Nat.divisorsAntidiagonal` without a special case for `n = 0`. -/
  let divisorsAntidiagonal (n : ℕ+) : Finset (ℕ+ × ℕ+) :=
    (Nat.divisorsAntidiagonal n).attach.map
      ⟨fun x =>
        (⟨x.val.1, Nat.pos_of_mem_divisors <| Nat.fst_mem_divisors_of_mem_antidiagonal x.prop⟩,
        ⟨x.val.2, Nat.pos_of_mem_divisors <| Nat.snd_mem_divisors_of_mem_antidiagonal x.prop⟩),
      fun _ _ h => Subtype.ext <| Prod.ext (congr_arg (·.1.val) h) (congr_arg (·.2.val) h)⟩

  have mem_divisorsAntidiagonal {n : ℕ+} (x : ℕ+ × ℕ+) :
    x ∈ divisorsAntidiagonal n ↔ x.1 * x.2 = n := by
    simp_rw [divisorsAntidiagonal, Finset.mem_map, Finset.mem_attach, Function.Embedding.coeFn_mk,
      Prod.ext_iff, true_and, ← coe_inj, Subtype.exists]
    /-
      divisorsAntidiagonal : PNat → Finset (Prod PNat PNat) := fun n => Finset.map { …
      n : PNat
      x : Prod PNat PNat
      ⊢ Iff (Exists fun a => Exists fun h => And (Eq ↑⟨a.1, ⋯⟩ ↑x.1) (Eq ↑⟨a.2, ⋯⟩ ↑ …
    -/
    aesop
    /-
      🎉 no goals
    -/
  { antidiagonal := fun n ↦ divisorsAntidiagonal (Additive.toMul n) |>.map
      (.prodMap (Additive.ofMul.toEmbedding) (Additive.ofMul.toEmbedding))
                           /-
                             divisorsAntidiagonal : PNat → Finset (Prod PNat PNat) := fun n => Finset.map { …
                             mem_divisorsAntidiagonal : ∀ {n : PNat} (x : Prod PNat PNat), Iff (Membership. …
                             ⊢ ∀ {n : Additive PNat} {a : Prod (Additive PNat) (Additive PNat)}, Iff (Membe …
                           -/
    mem_antidiagonal := by simp [← ofMul_mul, mem_divisorsAntidiagonal] }
                           /-
                             🎉 no goals
                           -/


/-- The `Finset` of all `d`-tuples of natural numbers whose product is `n`. Defined to be `∅` when
`n = 0`. -/
def finMulAntidiag (d : ℕ) (n : ℕ) : Finset (Fin d → ℕ) :=
  if hn : 0 < n then
    (Finset.finAntidiagonal d (Additive.ofMul (α := ℕ+) ⟨n, hn⟩)).map <|
      .arrowCongrRight <| Additive.toMul.toEmbedding.trans <| ⟨PNat.val, PNat.coe_injective⟩
  else
    ∅


@[simp]
theorem mem_finMulAntidiag {d n : ℕ} {f : Fin d → ℕ} :
    f ∈ finMulAntidiag d n ↔ ∏ i, f i = n ∧ n ≠ 0 := by
  /-
    d n : Nat
    f : Fin d → Nat
    ⊢ Iff (Membership.mem (d.finMulAntidiag n) f) (And (Eq (Finset.univ.prod fun i …
  -/
  unfold finMulAntidiag
  /-
    d n : Nat
    f : Fin d → Nat
    ⊢ Iff (Membership.mem (dite (LT.lt 0 n) (fun hn => Finset.map (Additive.toMul. …
  -/
  split_ifs with h
  · simp_rw [mem_map, mem_finAntidiagonal, Function.Embedding.arrowCongrRight_apply,
      Function.comp_def, Function.Embedding.trans_apply, Equiv.coe_toEmbedding,
      Function.Embedding.coeFn_mk, ← Additive.ofMul.symm_apply_eq, Additive.ofMul_symm_eq,
      toMul_sum, (Equiv.piCongrRight fun _=> Additive.ofMul).surjective.exists,
      Equiv.piCongrRight_apply, Pi.map_apply, toMul_ofMul, ← PNat.coe_inj, PNat.mk_coe,
      PNat.coe_prod]
    /-
      case pos
      d n : Nat
      f : Fin d → Nat
      h : LT.lt 0 n
      ⊢ Iff (Exists fun x => And (Eq (Finset.univ.prod fun i => ↑(x i)) n) (Eq (fun  …
    -/
    constructor
      /-
        case pos.mp
        d n : Nat
        f : Fin d → Nat
        h : LT.lt 0 n
        ⊢ (Exists fun x => And (Eq (Finset.univ.prod fun i => ↑(x i)) n) (Eq (fun i => …
      -/
    · rintro ⟨a, ha_mem, rfl⟩
      /-
        case pos.mp.intro.intro
        d n : Nat
        h : LT.lt 0 n
        a : Fin d → PNat
        ha_mem : Eq (Finset.univ.prod fun i => ↑(a i)) n
        ⊢ And (Eq (Finset.univ.prod fun i => (fun i => ↑(a i)) i) n) (Ne n 0)
      -/
      exact ⟨ha_mem, h.ne.symm⟩
      /-
        🎉 no goals
      -/
      /-
        case pos.mpr
        d n : Nat
        f : Fin d → Nat
        h : LT.lt 0 n
        ⊢ And (Eq (Finset.univ.prod fun i => f i) n) (Ne n 0) → Exists fun x => And (E …
      -/
    · rintro ⟨rfl, _⟩
      /-
        case pos.mpr.intro
        d : Nat
        f : Fin d → Nat
        h : LT.lt 0 (Finset.univ.prod fun i => f i)
        right✝ : Ne (Finset.univ.prod fun i => f i) 0
        ⊢ Exists fun x => And (Eq (Finset.univ.prod fun i => ↑(x i)) (Finset.univ.prod …
      -/
      refine ⟨fun i ↦ ⟨f i, ?_⟩, rfl, funext fun _ => rfl⟩
      /-
        case pos.mpr.intro
        d : Nat
        f : Fin d → Nat
        h : LT.lt 0 (Finset.univ.prod fun i => f i)
        right✝ : Ne (Finset.univ.prod fun i => f i) 0
        i : Fin d
        ⊢ LT.lt 0 (f i)
      -/
      apply Nat.pos_of_ne_zero
      /-
        case pos.mpr.intro.a
        d : Nat
        f : Fin d → Nat
        h : LT.lt 0 (Finset.univ.prod fun i => f i)
        right✝ : Ne (Finset.univ.prod fun i => f i) 0
        i : Fin d
        ⊢ Ne (f i) 0
      -/
      exact Finset.prod_ne_zero_iff.mp h.ne.symm _ (mem_univ _)
      /-
        🎉 no goals
      -/
    /-
      case neg
      d n : Nat
      f : Fin d → Nat
      h : Not (LT.lt 0 n)
      ⊢ Iff (Membership.mem EmptyCollection.emptyCollection f) (And (Eq (Finset.univ …
    -/
  · simp only [not_lt, nonpos_iff_eq_zero] at h
    /-
      case neg
      d n : Nat
      f : Fin d → Nat
      h : Eq n 0
      ⊢ Iff (Membership.mem EmptyCollection.emptyCollection f) (And (Eq (Finset.univ …
    -/
    simp only [h, not_mem_empty, ne_eq, not_true_eq_false, and_false]
    /-
      🎉 no goals
    -/


@[simp]
theorem finMulAntidiag_zero_right (d : ℕ) :
    finMulAntidiag d 0 = ∅ := rfl


theorem finMulAntidiag_one {d : ℕ} :
    finMulAntidiag d 1 = {fun _ => 1} := by
  /-
    d : Nat
    ⊢ Eq (d.finMulAntidiag 1) (Singleton.singleton fun x => 1)
  -/
  ext f
  /-
    case h
    d : Nat
    f : Fin d → Nat
    ⊢ Iff (Membership.mem (d.finMulAntidiag 1) f) (Membership.mem (Singleton.singl …
  -/
  simp only [mem_finMulAntidiag, and_true, mem_singleton]
  /-
    case h
    d : Nat
    f : Fin d → Nat
    ⊢ Iff (And (Eq (Finset.univ.prod fun i => f i) 1) (Ne 1 0)) (Eq f fun x => 1)
  -/
  constructor
    /-
      case h.mp
      d : Nat
      f : Fin d → Nat
      ⊢ And (Eq (Finset.univ.prod fun i => f i) 1) (Ne 1 0) → Eq f fun x => 1
    -/
  · intro ⟨hf, _⟩; ext i
    /-
      case h.mp.h
      d : Nat
      f : Fin d → Nat
      hf : Eq (Finset.univ.prod fun i => f i) 1
      right✝ : Ne 1 0
      i : Fin d
      ⊢ Eq (f i) 1
    -/
    rw [← Nat.dvd_one, ← hf]
    /-
      case h.mp.h
      d : Nat
      f : Fin d → Nat
      hf : Eq (Finset.univ.prod fun i => f i) 1
      right✝ : Ne 1 0
      i : Fin d
      ⊢ Dvd.dvd (f i) (Finset.univ.prod fun i => f i)
    -/
    exact dvd_prod_of_mem f (mem_univ _)
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      d : Nat
      f : Fin d → Nat
      ⊢ (Eq f fun x => 1) → And (Eq (Finset.univ.prod fun i => f i) 1) (Ne 1 0)
    -/
  · rintro rfl
    simp only [prod_const_one, implies_true, ne_eq, one_ne_zero, not_false_eq_true,
    and_self]


theorem finMulAntidiag_zero_left {n : ℕ} (hn : n ≠ 1) :
    finMulAntidiag 0 n = ∅ := by
  /-
    n : Nat
    hn : Ne n 1
    ⊢ Eq (Nat.finMulAntidiag 0 n) EmptyCollection.emptyCollection
  -/
  ext
  /-
    case h
    n : Nat
    hn : Ne n 1
    a✝ : Fin 0 → Nat
    ⊢ Iff (Membership.mem (Nat.finMulAntidiag 0 n) a✝) (Membership.mem EmptyCollec …
  -/
  simp [hn.symm]
  /-
    🎉 no goals
  -/


theorem dvd_of_mem_finMulAntidiag {n d : ℕ} {f : Fin d → ℕ} (hf : f ∈ finMulAntidiag d n)
    (i : Fin d) : f i ∣ n := by
  /-
    n d : Nat
    f : Fin d → Nat
    hf : Membership.mem (d.finMulAntidiag n) f
    i : Fin d
    ⊢ Dvd.dvd (f i) n
  -/
  rw [mem_finMulAntidiag] at hf
  /-
    n d : Nat
    f : Fin d → Nat
    hf : And (Eq (Finset.univ.prod fun i => f i) n) (Ne n 0)
    i : Fin d
    ⊢ Dvd.dvd (f i) n
  -/
  rw [← hf.1]
  /-
    n d : Nat
    f : Fin d → Nat
    hf : And (Eq (Finset.univ.prod fun i => f i) n) (Ne n 0)
    i : Fin d
    ⊢ Dvd.dvd (f i) (Finset.univ.prod fun i => f i)
  -/
  exact dvd_prod_of_mem f (mem_univ i)
  /-
    🎉 no goals
  -/


theorem ne_zero_of_mem_finMulAntidiag {d n : ℕ} {f : Fin d → ℕ}
    (hf : f ∈ finMulAntidiag d n) (i : Fin d) : f i ≠ 0 :=
  ne_zero_of_dvd_ne_zero (mem_finMulAntidiag.mp hf).2 (dvd_of_mem_finMulAntidiag hf i)


theorem prod_eq_of_mem_finMulAntidiag {d n : ℕ} {f : Fin d → ℕ}
    (hf : f ∈ finMulAntidiag d n) : ∏ i, f i = n :=
  (mem_finMulAntidiag.mp hf).1


theorem finMulAntidiag_eq_piFinset_divisors_filter {d m n : ℕ} (hmn : m ∣ n) (hn : n ≠ 0) :
    finMulAntidiag d m =
      {f ∈ Fintype.piFinset fun _ : Fin d => n.divisors | ∏ i, f i = m} := by
  /-
    d m n : Nat
    hmn : Dvd.dvd m n
    hn : Ne n 0
    ⊢ Eq (d.finMulAntidiag m) (Finset.filter (fun f => Eq (Finset.univ.prod fun i  …
  -/
  ext f
  simp only [mem_univ, not_true, IsEmpty.forall_iff, implies_true, ne_eq, true_and,
    Fintype.mem_piFinset, mem_divisors, Nat.isUnit_iff, mem_filter]
  /-
    case h
    d m n : Nat
    hmn : Dvd.dvd m n
    hn : Ne n 0
    f : Fin d → Nat
    ⊢ Iff (Membership.mem (d.finMulAntidiag m) f) (And (∀ (a : Fin d), And (Dvd.dv …
  -/
  constructor
    /-
      case h.mp
      d m n : Nat
      hmn : Dvd.dvd m n
      hn : Ne n 0
      f : Fin d → Nat
      ⊢ Membership.mem (d.finMulAntidiag m) f → And (∀ (a : Fin d), And (Dvd.dvd (f  …
    -/
  · intro hf
    /-
      case h.mp
      d m n : Nat
      hmn : Dvd.dvd m n
      hn : Ne n 0
      f : Fin d → Nat
      hf : Membership.mem (d.finMulAntidiag m) f
      ⊢ And (∀ (a : Fin d), And (Dvd.dvd (f a) n) (Not (Eq n 0))) (Eq (Finset.univ.p …
    -/
    refine ⟨?_, prod_eq_of_mem_finMulAntidiag hf⟩
    /-
      case h.mp
      d m n : Nat
      hmn : Dvd.dvd m n
      hn : Ne n 0
      f : Fin d → Nat
      hf : Membership.mem (d.finMulAntidiag m) f
      ⊢ ∀ (a : Fin d), And (Dvd.dvd (f a) n) (Not (Eq n 0))
    -/
    exact fun i => ⟨(dvd_of_mem_finMulAntidiag hf i).trans hmn, hn⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      d m n : Nat
      hmn : Dvd.dvd m n
      hn : Ne n 0
      f : Fin d → Nat
      ⊢ And (∀ (a : Fin d), And (Dvd.dvd (f a) n) (Not (Eq n 0))) (Eq (Finset.univ.p …
    -/
  · rw [mem_finMulAntidiag]
    /-
      case h.mpr
      d m n : Nat
      hmn : Dvd.dvd m n
      hn : Ne n 0
      f : Fin d → Nat
      ⊢ And (∀ (a : Fin d), And (Dvd.dvd (f a) n) (Not (Eq n 0))) (Eq (Finset.univ.p …
    -/
    exact fun ⟨_, hprod⟩ => ⟨hprod, ne_zero_of_dvd_ne_zero hn hmn⟩
    /-
      🎉 no goals
    -/


lemma image_apply_finMulAntidiag {d n : ℕ} {i : Fin d} (hd : d ≠ 1) :
    (finMulAntidiag d n).image (fun f => f i) = divisors n := by
  /-
    d n : Nat
    i : Fin d
    hd : Ne d 1
    ⊢ Eq (Finset.image (fun f => f i) (d.finMulAntidiag n)) n.divisors
  -/
  ext k
  /-
    case h
    d n : Nat
    i : Fin d
    hd : Ne d 1
    k : Nat
    ⊢ Iff (Membership.mem (Finset.image (fun f => f i) (d.finMulAntidiag n)) k) (M …
  -/
  simp only [mem_image, ne_eq, mem_divisors, Nat.isUnit_iff]
  /-
    case h
    d n : Nat
    i : Fin d
    hd : Ne d 1
    k : Nat
    ⊢ Iff (Exists fun a => And (Membership.mem (d.finMulAntidiag n) a) (Eq (a i) k …
  -/
  constructor
    /-
      case h.mp
      d n : Nat
      i : Fin d
      hd : Ne d 1
      k : Nat
      ⊢ (Exists fun a => And (Membership.mem (d.finMulAntidiag n) a) (Eq (a i) k)) → …
    -/
  · rintro ⟨f, hf, rfl⟩
    /-
      case h.mp.intro.intro
      d n : Nat
      i : Fin d
      hd : Ne d 1
      f : Fin d → Nat
      hf : Membership.mem (d.finMulAntidiag n) f
      ⊢ And (Dvd.dvd (f i) n) (Not (Eq n 0))
    -/
    exact ⟨dvd_of_mem_finMulAntidiag hf _, (mem_finMulAntidiag.mp hf).2⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      d n : Nat
      i : Fin d
      hd : Ne d 1
      k : Nat
      ⊢ And (Dvd.dvd k n) (Not (Eq n 0)) → Exists fun a => And (Membership.mem (d.fi …
    -/
  · simp_rw [mem_finMulAntidiag]
    /-
      case h.mpr
      d n : Nat
      i : Fin d
      hd : Ne d 1
      k : Nat
      ⊢ And (Dvd.dvd k n) (Not (Eq n 0)) → Exists fun a => And (And (Eq (Finset.univ …
    -/
    rintro ⟨⟨r, rfl⟩, hn⟩
    have hs : Nontrivial (Fin d) := by
      rw [Fin.nontrivial_iff_two_le]
      obtain rfl | hd' := eq_or_ne d 0
      · exact i.elim0
      omega
    /-
      case h.mpr.intro.intro
      d : Nat
      i : Fin d
      hd : Ne d 1
      k r : Nat
      hn : Not (Eq (HMul.hMul k r) 0)
      hs : Nontrivial (Fin d)
      ⊢ Exists fun a => And (And (Eq (Finset.univ.prod fun i => a i) (HMul.hMul k r) …
    -/
    obtain ⟨i', hi_ne⟩ := exists_ne i
    /-
      case h.mpr.intro.intro.intro
      d : Nat
      i : Fin d
      hd : Ne d 1
      k r : Nat
      hn : Not (Eq (HMul.hMul k r) 0)
      hs : Nontrivial (Fin d)
      i' : Fin d
      hi_ne : Ne i' i
      ⊢ Exists fun a => And (And (Eq (Finset.univ.prod fun i => a i) (HMul.hMul k r) …
    -/
    use fun j => if j = i then k else if j = i' then r else 1
    /-
      case h
      d : Nat
      i : Fin d
      hd : Ne d 1
      k r : Nat
      hn : Not (Eq (HMul.hMul k r) 0)
      hs : Nontrivial (Fin d)
      i' : Fin d
      hi_ne : Ne i' i
      ⊢ And (And (Eq (Finset.univ.prod fun i_1 => (fun j => ite (Eq j i) k (ite (Eq  …
    -/
    simp only [ite_true, and_true, hn]
    rw [← Finset.mul_prod_erase (a:=i) (h:=mem_univ _),
      ← Finset.mul_prod_erase (a:= i')]
      /-
        case h
        d : Nat
        i : Fin d
        hd : Ne d 1
        k r : Nat
        hn : Not (Eq (HMul.hMul k r) 0)
        hs : Nontrivial (Fin d)
        i' : Fin d
        hi_ne : Ne i' i
        ⊢ And (Eq (HMul.hMul (ite (Eq i i) k (ite (Eq i i') r 1)) (HMul.hMul (ite (Eq  …
      -/
    · rw [if_neg hi_ne, if_pos rfl, if_pos rfl, prod_eq_one]
        /-
          case h
          d : Nat
          i : Fin d
          hd : Ne d 1
          k r : Nat
          hn : Not (Eq (HMul.hMul k r) 0)
          hs : Nontrivial (Fin d)
          i' : Fin d
          hi_ne : Ne i' i
          ⊢ And (Eq (HMul.hMul k (HMul.hMul r 1)) (HMul.hMul k r)) (Ne (HMul.hMul k r) 0)
        -/
      · refine ⟨by ring, hn⟩
        /-
          🎉 no goals
        -/
      /-
        case h
        d : Nat
        i : Fin d
        hd : Ne d 1
        k r : Nat
        hn : Not (Eq (HMul.hMul k r) 0)
        hs : Nontrivial (Fin d)
        i' : Fin d
        hi_ne : Ne i' i
        ⊢ ∀ (x : Fin d), Membership.mem ((Finset.univ.erase i).erase i') x → Eq (ite ( …
      -/
      intro j hj
      /-
        case h
        d : Nat
        i : Fin d
        hd : Ne d 1
        k r : Nat
        hn : Not (Eq (HMul.hMul k r) 0)
        hs : Nontrivial (Fin d)
        i' : Fin d
        hi_ne : Ne i' i
        j : Fin d
        hj : Membership.mem ((Finset.univ.erase i).erase i') j
        ⊢ Eq (ite (Eq j i) k (ite (Eq j i') r 1)) 1
      -/
      simp only [mem_erase, ne_eq, mem_univ, and_true] at hj
      /-
        case h
        d : Nat
        i : Fin d
        hd : Ne d 1
        k r : Nat
        hn : Not (Eq (HMul.hMul k r) 0)
        hs : Nontrivial (Fin d)
        i' : Fin d
        hi_ne : Ne i' i
        j : Fin d
        hj : And (Not (Eq j i')) (Not (Eq j i))
        ⊢ Eq (ite (Eq j i) k (ite (Eq j i') r 1)) 1
      -/
      rw [if_neg hj.1, if_neg hj.2]
      /-
        🎉 no goals
      -/
    /-
      case h.h
      d : Nat
      i : Fin d
      hd : Ne d 1
      k r : Nat
      hn : Not (Eq (HMul.hMul k r) 0)
      hs : Nontrivial (Fin d)
      i' : Fin d
      hi_ne : Ne i' i
      ⊢ Membership.mem (Finset.univ.erase i) i'
    -/
    exact mem_erase.mpr ⟨hi_ne, mem_univ _⟩
    /-
      🎉 no goals
    -/


lemma image_piFinTwoEquiv_finMulAntidiag {n : ℕ} :
    (finMulAntidiag 2 n).image (piFinTwoEquiv <| fun _ => ℕ) = divisorsAntidiagonal n := by
  /-
    n : Nat
    ⊢ Eq (Finset.image (⇑(piFinTwoEquiv fun x => Nat)) (Nat.finMulAntidiag 2 n)) n …
  -/
  ext x
  /-
    case h
    n : Nat
    x : Prod Nat Nat
    ⊢ Iff (Membership.mem (Finset.image (⇑(piFinTwoEquiv fun x => Nat)) (Nat.finMu …
  -/
  simp [(piFinTwoEquiv <| fun _ => ℕ).symm.surjective.exists]
  /-
    🎉 no goals
  -/


lemma finMulAntidiag_existsUnique_prime_dvd {d n p : ℕ} (hn : Squarefree n)
    (hp : p ∈ n.primeFactorsList) (f : Fin d → ℕ) (hf : f ∈ finMulAntidiag d n) :
    ∃! i, p ∣ f i := by
  /-
    d n p : Nat
    hn : Squarefree n
    hp : Membership.mem n.primeFactorsList p
    f : Fin d → Nat
    hf : Membership.mem (d.finMulAntidiag n) f
    ⊢ ExistsUnique fun i => Dvd.dvd p (f i)
  -/
  rw [mem_finMulAntidiag] at hf
  /-
    d n p : Nat
    hn : Squarefree n
    hp : Membership.mem n.primeFactorsList p
    f : Fin d → Nat
    hf : And (Eq (Finset.univ.prod fun i => f i) n) (Ne n 0)
    ⊢ ExistsUnique fun i => Dvd.dvd p (f i)
  -/
  rw [mem_primeFactorsList hf.2, ← hf.1, hp.1.prime.dvd_finset_prod_iff] at hp
  /-
    d n p : Nat
    hn : Squarefree n
    f : Fin d → Nat
    hp : And (Nat.Prime p) (Exists fun a => And (Membership.mem Finset.univ a) (Dv …
    hf : And (Eq (Finset.univ.prod fun i => f i) n) (Ne n 0)
    ⊢ ExistsUnique fun i => Dvd.dvd p (f i)
  -/
  obtain ⟨i, his, hi⟩ := hp.2
  /-
    case intro.intro
    d n p : Nat
    hn : Squarefree n
    f : Fin d → Nat
    hp : And (Nat.Prime p) (Exists fun a => And (Membership.mem Finset.univ a) (Dv …
    hf : And (Eq (Finset.univ.prod fun i => f i) n) (Ne n 0)
    i : Fin d
    his : Membership.mem Finset.univ i
    hi : Dvd.dvd p (f i)
    ⊢ ExistsUnique fun i => Dvd.dvd p (f i)
  -/
  refine ⟨i, hi, ?_⟩
  /-
    case intro.intro
    d n p : Nat
    hn : Squarefree n
    f : Fin d → Nat
    hp : And (Nat.Prime p) (Exists fun a => And (Membership.mem Finset.univ a) (Dv …
    hf : And (Eq (Finset.univ.prod fun i => f i) n) (Ne n 0)
    i : Fin d
    his : Membership.mem Finset.univ i
    hi : Dvd.dvd p (f i)
    ⊢ ∀ (y : Fin d), (fun i => Dvd.dvd p (f i)) y → Eq y i
  -/
  intro j hj
  /-
    case intro.intro
    d n p : Nat
    hn : Squarefree n
    f : Fin d → Nat
    hp : And (Nat.Prime p) (Exists fun a => And (Membership.mem Finset.univ a) (Dv …
    hf : And (Eq (Finset.univ.prod fun i => f i) n) (Ne n 0)
    i : Fin d
    his : Membership.mem Finset.univ i
    hi : Dvd.dvd p (f i)
    j : Fin d
    hj : Dvd.dvd p (f j)
    ⊢ Eq j i
  -/
  by_contra hij
  /-
    case intro.intro
    d n p : Nat
    hn : Squarefree n
    f : Fin d → Nat
    hp : And (Nat.Prime p) (Exists fun a => And (Membership.mem Finset.univ a) (Dv …
    hf : And (Eq (Finset.univ.prod fun i => f i) n) (Ne n 0)
    i : Fin d
    his : Membership.mem Finset.univ i
    hi : Dvd.dvd p (f i)
    j : Fin d
    hj : Dvd.dvd p (f j)
    hij : Not (Eq j i)
    ⊢ False
  -/
  apply Nat.Prime.not_coprime_iff_dvd.mpr ⟨p, hp.1, hi, hj⟩
  /-
    case intro.intro
    d n p : Nat
    hn : Squarefree n
    f : Fin d → Nat
    hp : And (Nat.Prime p) (Exists fun a => And (Membership.mem Finset.univ a) (Dv …
    hf : And (Eq (Finset.univ.prod fun i => f i) n) (Ne n 0)
    i : Fin d
    his : Membership.mem Finset.univ i
    hi : Dvd.dvd p (f i)
    j : Fin d
    hj : Dvd.dvd p (f j)
    hij : Not (Eq j i)
    ⊢ (f i).Coprime (f j)
  -/
  apply Nat.coprime_of_squarefree_mul
  /-
    case intro.intro.h
    d n p : Nat
    hn : Squarefree n
    f : Fin d → Nat
    hp : And (Nat.Prime p) (Exists fun a => And (Membership.mem Finset.univ a) (Dv …
    hf : And (Eq (Finset.univ.prod fun i => f i) n) (Ne n 0)
    i : Fin d
    his : Membership.mem Finset.univ i
    hi : Dvd.dvd p (f i)
    j : Fin d
    hj : Dvd.dvd p (f j)
    hij : Not (Eq j i)
    ⊢ Squarefree (HMul.hMul (f i) (f j))
  -/
  apply hn.squarefree_of_dvd
  rw [← hf.1, ← Finset.mul_prod_erase _ _ (his),
    ← Finset.mul_prod_erase _ _ (mem_erase.mpr ⟨hij, mem_univ _⟩), ← mul_assoc]
  /-
    case intro.intro.h
    d n p : Nat
    hn : Squarefree n
    f : Fin d → Nat
    hp : And (Nat.Prime p) (Exists fun a => And (Membership.mem Finset.univ a) (Dv …
    hf : And (Eq (Finset.univ.prod fun i => f i) n) (Ne n 0)
    i : Fin d
    his : Membership.mem Finset.univ i
    hi : Dvd.dvd p (f i)
    j : Fin d
    hj : Dvd.dvd p (f j)
    hij : Not (Eq j i)
    ⊢ Dvd.dvd (HMul.hMul (f i) (f j)) (HMul.hMul (HMul.hMul (f i) (f j)) (((Finset …
  -/
  apply Nat.dvd_mul_right
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-17")]
alias finMulAntidiag_exists_unique_prime_dvd := finMulAntidiag_existsUnique_prime_dvd


private def primeFactorsPiBij (d n : ℕ) :
    ∀ f ∈ (n.primeFactors.pi fun _ => (univ : Finset <| Fin d)), Fin d → ℕ :=
  fun f _ i => ∏ p ∈ {p ∈ n.primeFactors.attach | f p.1 p.2 = i} , p


private theorem primeFactorsPiBij_img (d n : ℕ) (hn : Squarefree n)
  (f : (p : ℕ) → p ∈ n.primeFactors → Fin d) (hf : f ∈ pi n.primeFactors fun _ => univ) :
    Nat.primeFactorsPiBij d n f hf ∈ finMulAntidiag d n := by
  /-
    d n : Nat
    hn : Squarefree n
    f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
    hf : Membership.mem (n.primeFactors.pi fun x => Finset.univ) f
    ⊢ Membership.mem (d.finMulAntidiag n) (Nat.primeFactorsPiBij d n f hf)
  -/
  rw [mem_finMulAntidiag]
  /-
    d n : Nat
    hn : Squarefree n
    f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
    hf : Membership.mem (n.primeFactors.pi fun x => Finset.univ) f
    ⊢ And (Eq (Finset.univ.prod fun i => Nat.primeFactorsPiBij d n f hf i) n) (Ne  …
  -/
  refine ⟨?_, hn.ne_zero⟩
  /-
    d n : Nat
    hn : Squarefree n
    f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
    hf : Membership.mem (n.primeFactors.pi fun x => Finset.univ) f
    ⊢ Eq (Finset.univ.prod fun i => Nat.primeFactorsPiBij d n f hf i) n
  -/
  unfold Nat.primeFactorsPiBij
  /-
    d n : Nat
    hn : Squarefree n
    f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
    hf : Membership.mem (n.primeFactors.pi fun x => Finset.univ) f
    ⊢ Eq (Finset.univ.prod fun i => (Finset.filter (fun p => Eq (f ↑p ⋯) i) n.prim …
  -/
  rw [prod_fiberwise_of_maps_to, prod_attach (f := fun x => x)]
    /-
      d n : Nat
      hn : Squarefree n
      f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
      hf : Membership.mem (n.primeFactors.pi fun x => Finset.univ) f
      ⊢ Eq (n.primeFactors.prod fun x => x) n
    -/
  · apply prod_primeFactors_of_squarefree hn
    /-
      🎉 no goals
    -/
    /-
      case h
      d n : Nat
      hn : Squarefree n
      f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
      hf : Membership.mem (n.primeFactors.pi fun x => Finset.univ) f
      ⊢ ∀ (i : Subtype fun x => Membership.mem n.primeFactors x), Membership.mem n.p …
    -/
  · apply fun _ _ => mem_univ _
    /-
      🎉 no goals
    -/


private theorem primeFactorsPiBij_inj (d n : ℕ)
    (f : (p : ℕ) → p ∈ n.primeFactors → Fin d) (hf : f ∈ pi n.primeFactors fun _ => univ)
    (g : (p : ℕ) → p ∈ n.primeFactors → Fin d) (hg : g ∈ pi n.primeFactors fun _ => univ) :
    Nat.primeFactorsPiBij d n f hf = Nat.primeFactorsPiBij d n g hg → f = g := by
  /-
    d n : Nat
    f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
    hf : Membership.mem (n.primeFactors.pi fun x => Finset.univ) f
    g : (p : Nat) → Membership.mem n.primeFactors p → Fin d
    hg : Membership.mem (n.primeFactors.pi fun x => Finset.univ) g
    ⊢ Eq (Nat.primeFactorsPiBij d n f hf) (Nat.primeFactorsPiBij d n g hg) → Eq f g
  -/
  contrapose!
  /-
    d n : Nat
    f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
    hf : Membership.mem (n.primeFactors.pi fun x => Finset.univ) f
    g : (p : Nat) → Membership.mem n.primeFactors p → Fin d
    hg : Membership.mem (n.primeFactors.pi fun x => Finset.univ) g
    ⊢ Ne f g → Ne (Nat.primeFactorsPiBij d n f hf) (Nat.primeFactorsPiBij d n g hg)
  -/
  simp_rw [Function.ne_iff]
  /-
    d n : Nat
    f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
    hf : Membership.mem (n.primeFactors.pi fun x => Finset.univ) f
    g : (p : Nat) → Membership.mem n.primeFactors p → Fin d
    hg : Membership.mem (n.primeFactors.pi fun x => Finset.univ) g
    ⊢ (Exists fun a => Exists fun a_1 => Ne (f a a_1) (g a a_1)) → Exists fun a => …
  -/
  intro ⟨p, hp, hfg⟩
  /-
    d n : Nat
    f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
    hf : Membership.mem (n.primeFactors.pi fun x => Finset.univ) f
    g : (p : Nat) → Membership.mem n.primeFactors p → Fin d
    hg : Membership.mem (n.primeFactors.pi fun x => Finset.univ) g
    p : Nat
    hp : Membership.mem n.primeFactors p
    hfg : Ne (f p hp) (g p hp)
    ⊢ Exists fun a => Ne (Nat.primeFactorsPiBij d n f hf a) (Nat.primeFactorsPiBij …
  -/
  use f p hp
  /-
    case h
    d n : Nat
    f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
    hf : Membership.mem (n.primeFactors.pi fun x => Finset.univ) f
    g : (p : Nat) → Membership.mem n.primeFactors p → Fin d
    hg : Membership.mem (n.primeFactors.pi fun x => Finset.univ) g
    p : Nat
    hp : Membership.mem n.primeFactors p
    hfg : Ne (f p hp) (g p hp)
    ⊢ Ne (Nat.primeFactorsPiBij d n f hf (f p hp)) (Nat.primeFactorsPiBij d n g hg …
  -/
  dsimp only [Nat.primeFactorsPiBij]
  /-
    case h
    d n : Nat
    f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
    hf : Membership.mem (n.primeFactors.pi fun x => Finset.univ) f
    g : (p : Nat) → Membership.mem n.primeFactors p → Fin d
    hg : Membership.mem (n.primeFactors.pi fun x => Finset.univ) g
    p : Nat
    hp : Membership.mem n.primeFactors p
    hfg : Ne (f p hp) (g p hp)
    ⊢ Ne ((Finset.filter (fun p_1 => Eq (f ↑p_1 ⋯) (f p hp)) n.primeFactors.attach …
  -/
  apply ne_of_mem_of_not_mem (s := {x | p ∣ x}) <;> simp_rw [Set.mem_setOf_eq]
    /-
      case h.h
      d n : Nat
      f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
      hf : Membership.mem (n.primeFactors.pi fun x => Finset.univ) f
      g : (p : Nat) → Membership.mem n.primeFactors p → Fin d
      hg : Membership.mem (n.primeFactors.pi fun x => Finset.univ) g
      p : Nat
      hp : Membership.mem n.primeFactors p
      hfg : Ne (f p hp) (g p hp)
      ⊢ Dvd.dvd p ((Finset.filter (fun p_1 => Eq (f ↑p_1 ⋯) (f p hp)) n.primeFactors …
    -/
  · rw [Finset.prod_filter]
    /-
      case h.h
      d n : Nat
      f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
      hf : Membership.mem (n.primeFactors.pi fun x => Finset.univ) f
      g : (p : Nat) → Membership.mem n.primeFactors p → Fin d
      hg : Membership.mem (n.primeFactors.pi fun x => Finset.univ) g
      p : Nat
      hp : Membership.mem n.primeFactors p
      hfg : Ne (f p hp) (g p hp)
      ⊢ Dvd.dvd p (n.primeFactors.attach.prod fun a => ite (Eq (f ↑a ⋯) (f p hp)) (↑ …
    -/
    convert Finset.dvd_prod_of_mem _ (mem_attach (n.primeFactors) ⟨p, hp⟩)
    /-
      case h.e'_3
      d n : Nat
      f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
      hf : Membership.mem (n.primeFactors.pi fun x => Finset.univ) f
      g : (p : Nat) → Membership.mem n.primeFactors p → Fin d
      hg : Membership.mem (n.primeFactors.pi fun x => Finset.univ) g
      p : Nat
      hp : Membership.mem n.primeFactors p
      hfg : Ne (f p hp) (g p hp)
      ⊢ Eq p (ite (Eq (f ↑⟨p, hp⟩ ⋯) (f p hp)) (↑⟨p, hp⟩) 1)
    -/
    rw [if_pos rfl]
    /-
      🎉 no goals
    -/
    /-
      case h.a
      d n : Nat
      f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
      hf : Membership.mem (n.primeFactors.pi fun x => Finset.univ) f
      g : (p : Nat) → Membership.mem n.primeFactors p → Fin d
      hg : Membership.mem (n.primeFactors.pi fun x => Finset.univ) g
      p : Nat
      hp : Membership.mem n.primeFactors p
      hfg : Ne (f p hp) (g p hp)
      ⊢ Not (Dvd.dvd p ((Finset.filter (fun p_1 => Eq (g ↑p_1 ⋯) (f p hp)) n.primeFa …
    -/
  · rw [mem_primeFactors] at hp
    /-
      case h.a
      d n : Nat
      f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
      hf : Membership.mem (n.primeFactors.pi fun x => Finset.univ) f
      g : (p : Nat) → Membership.mem n.primeFactors p → Fin d
      hg : Membership.mem (n.primeFactors.pi fun x => Finset.univ) g
      p : Nat
      hp✝ : Membership.mem n.primeFactors p
      hp : And (Nat.Prime p) (And (Dvd.dvd p n) (Ne n 0))
      hfg : Ne (f p hp✝) (g p hp✝)
      ⊢ Not (Dvd.dvd p ((Finset.filter (fun p_1 => Eq (g ↑p_1 ⋯) (f p hp✝)) n.primeF …
    -/
    rw [Prime.dvd_finset_prod_iff hp.1.prime]
    /-
      case h.a
      d n : Nat
      f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
      hf : Membership.mem (n.primeFactors.pi fun x => Finset.univ) f
      g : (p : Nat) → Membership.mem n.primeFactors p → Fin d
      hg : Membership.mem (n.primeFactors.pi fun x => Finset.univ) g
      p : Nat
      hp✝ : Membership.mem n.primeFactors p
      hp : And (Nat.Prime p) (And (Dvd.dvd p n) (Ne n 0))
      hfg : Ne (f p hp✝) (g p hp✝)
      ⊢ Not (Exists fun a => And (Membership.mem (Finset.filter (fun p_1 => Eq (g ↑p …
    -/
    push_neg
    /-
      case h.a
      d n : Nat
      f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
      hf : Membership.mem (n.primeFactors.pi fun x => Finset.univ) f
      g : (p : Nat) → Membership.mem n.primeFactors p → Fin d
      hg : Membership.mem (n.primeFactors.pi fun x => Finset.univ) g
      p : Nat
      hp✝ : Membership.mem n.primeFactors p
      hp : And (Nat.Prime p) (And (Dvd.dvd p n) (Ne n 0))
      hfg : Ne (f p hp✝) (g p hp✝)
      ⊢ ∀ (a : Subtype fun x => Membership.mem n.primeFactors x), Membership.mem (Fi …
    -/
    intro q hq
    rw [Nat.prime_dvd_prime_iff_eq hp.1 (Nat.prime_of_mem_primeFactorsList
      <| List.mem_toFinset.mp q.2)]
    /-
      case h.a
      d n : Nat
      f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
      hf : Membership.mem (n.primeFactors.pi fun x => Finset.univ) f
      g : (p : Nat) → Membership.mem n.primeFactors p → Fin d
      hg : Membership.mem (n.primeFactors.pi fun x => Finset.univ) g
      p : Nat
      hp✝ : Membership.mem n.primeFactors p
      hp : And (Nat.Prime p) (And (Dvd.dvd p n) (Ne n 0))
      hfg : Ne (f p hp✝) (g p hp✝)
      q : Subtype fun x => Membership.mem n.primeFactors x
      hq : Membership.mem (Finset.filter (fun p_1 => Eq (g ↑p_1 ⋯) (f p hp✝)) n.prim …
      ⊢ Not (Eq p ↑q)
    -/
    rintro rfl
    /-
      case h.a
      d n : Nat
      f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
      hf : Membership.mem (n.primeFactors.pi fun x => Finset.univ) f
      g : (p : Nat) → Membership.mem n.primeFactors p → Fin d
      hg : Membership.mem (n.primeFactors.pi fun x => Finset.univ) g
      q : Subtype fun x => Membership.mem n.primeFactors x
      hp✝ : Membership.mem n.primeFactors ↑q
      hp : And (Nat.Prime ↑q) (And (Dvd.dvd (↑q) n) (Ne n 0))
      hfg : Ne (f (↑q) hp✝) (g (↑q) hp✝)
      hq : Membership.mem (Finset.filter (fun p => Eq (g ↑p ⋯) (f (↑q) hp✝)) n.prime …
      ⊢ False
    -/
    rw [(mem_filter.mp hq).2] at hfg
    /-
      case h.a
      d n : Nat
      f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
      hf : Membership.mem (n.primeFactors.pi fun x => Finset.univ) f
      g : (p : Nat) → Membership.mem n.primeFactors p → Fin d
      hg : Membership.mem (n.primeFactors.pi fun x => Finset.univ) g
      q : Subtype fun x => Membership.mem n.primeFactors x
      hp✝ : Membership.mem n.primeFactors ↑q
      hp : And (Nat.Prime ↑q) (And (Dvd.dvd (↑q) n) (Ne n 0))
      hfg : Ne (f (↑q) hp✝) (f (↑q) hp✝)
      hq : Membership.mem (Finset.filter (fun p => Eq (g ↑p ⋯) (f (↑q) hp✝)) n.prime …
      ⊢ False
    -/
    exact hfg rfl
    /-
      🎉 no goals
    -/


private theorem primeFactorsPiBij_surj (d n : ℕ) (hn : Squarefree n)
    (t : Fin d → ℕ) (ht : t ∈ finMulAntidiag d n) : ∃ (g : _)
    (hg : g ∈ pi n.primeFactors fun _ => univ), Nat.primeFactorsPiBij d n g hg = t := by
  have existsUnique := fun (p : ℕ) (hp : p ∈ n.primeFactors) =>
    (finMulAntidiag_existsUnique_prime_dvd hn
      (mem_primeFactors_iff_mem_primeFactorsList.mp hp) t ht)
  /-
    d n : Nat
    hn : Squarefree n
    t : Fin d → Nat
    ht : Membership.mem (d.finMulAntidiag n) t
    existsUnique : ∀ (p : Nat), Membership.mem n.primeFactors p → ExistsUnique fun …
    ⊢ Exists fun g => Exists fun hg => Eq (Nat.primeFactorsPiBij d n g hg) t
  -/
  choose f hf hf_unique using existsUnique
  /-
    d n : Nat
    hn : Squarefree n
    t : Fin d → Nat
    ht : Membership.mem (d.finMulAntidiag n) t
    f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
    hf : ∀ (p : Nat) (hp : Membership.mem n.primeFactors p), (fun i => Dvd.dvd p ( …
    hf_unique : ∀ (p : Nat) (hp : Membership.mem n.primeFactors p) (y : Fin d), (f …
    ⊢ Exists fun g => Exists fun hg => Eq (Nat.primeFactorsPiBij d n g hg) t
  -/
  refine ⟨f, ?_, ?_⟩
    /-
      case refine_1
      d n : Nat
      hn : Squarefree n
      t : Fin d → Nat
      ht : Membership.mem (d.finMulAntidiag n) t
      f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
      hf : ∀ (p : Nat) (hp : Membership.mem n.primeFactors p), (fun i => Dvd.dvd p ( …
      hf_unique : ∀ (p : Nat) (hp : Membership.mem n.primeFactors p) (y : Fin d), (f …
      ⊢ Membership.mem (n.primeFactors.pi fun x => Finset.univ) f
    -/
  · simp only [mem_pi, mem_univ, forall_true_iff]
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    d n : Nat
    hn : Squarefree n
    t : Fin d → Nat
    ht : Membership.mem (d.finMulAntidiag n) t
    f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
    hf : ∀ (p : Nat) (hp : Membership.mem n.primeFactors p), (fun i => Dvd.dvd p ( …
    hf_unique : ∀ (p : Nat) (hp : Membership.mem n.primeFactors p) (y : Fin d), (f …
    ⊢ Eq (Nat.primeFactorsPiBij d n f ⋯) t
  -/
  funext i
  /-
    case refine_2.h
    d n : Nat
    hn : Squarefree n
    t : Fin d → Nat
    ht : Membership.mem (d.finMulAntidiag n) t
    f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
    hf : ∀ (p : Nat) (hp : Membership.mem n.primeFactors p), (fun i => Dvd.dvd p ( …
    hf_unique : ∀ (p : Nat) (hp : Membership.mem n.primeFactors p) (y : Fin d), (f …
    i : Fin d
    ⊢ Eq (Nat.primeFactorsPiBij d n f ⋯ i) (t i)
  -/
  have : t i ∣ n := dvd_of_mem_finMulAntidiag ht _
  /-
    case refine_2.h
    d n : Nat
    hn : Squarefree n
    t : Fin d → Nat
    ht : Membership.mem (d.finMulAntidiag n) t
    f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
    hf : ∀ (p : Nat) (hp : Membership.mem n.primeFactors p), (fun i => Dvd.dvd p ( …
    hf_unique : ∀ (p : Nat) (hp : Membership.mem n.primeFactors p) (y : Fin d), (f …
    i : Fin d
    this : Dvd.dvd (t i) n
    ⊢ Eq (Nat.primeFactorsPiBij d n f ⋯ i) (t i)
  -/
  trans (∏ p ∈ n.primeFactors.attach, if p.1 ∣ t i then p else 1)
    /-
      d n : Nat
      hn : Squarefree n
      t : Fin d → Nat
      ht : Membership.mem (d.finMulAntidiag n) t
      f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
      hf : ∀ (p : Nat) (hp : Membership.mem n.primeFactors p), (fun i => Dvd.dvd p ( …
      hf_unique : ∀ (p : Nat) (hp : Membership.mem n.primeFactors p) (y : Fin d), (f …
      i : Fin d
      this : Dvd.dvd (t i) n
      ⊢ Eq (Nat.primeFactorsPiBij d n f ⋯ i) (n.primeFactors.attach.prod fun p => it …
    -/
  · rw [Nat.primeFactorsPiBij, ← prod_filter]
    /-
      d n : Nat
      hn : Squarefree n
      t : Fin d → Nat
      ht : Membership.mem (d.finMulAntidiag n) t
      f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
      hf : ∀ (p : Nat) (hp : Membership.mem n.primeFactors p), (fun i => Dvd.dvd p ( …
      hf_unique : ∀ (p : Nat) (hp : Membership.mem n.primeFactors p) (y : Fin d), (f …
      i : Fin d
      this : Dvd.dvd (t i) n
      ⊢ Eq ((Finset.filter (fun p => Eq (f ↑p ⋯) i) n.primeFactors.attach).prod fun  …
    -/
    congr
    /-
      case e_s.e_p
      d n : Nat
      hn : Squarefree n
      t : Fin d → Nat
      ht : Membership.mem (d.finMulAntidiag n) t
      f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
      hf : ∀ (p : Nat) (hp : Membership.mem n.primeFactors p), (fun i => Dvd.dvd p ( …
      hf_unique : ∀ (p : Nat) (hp : Membership.mem n.primeFactors p) (y : Fin d), (f …
      i : Fin d
      this : Dvd.dvd (t i) n
      ⊢ Eq (fun p => Eq (f ↑p ⋯) i) fun a => Dvd.dvd (↑a) (t i)
    -/
    ext ⟨p, hp⟩
    /-
      case e_s.e_p.h.mk.a
      d n : Nat
      hn : Squarefree n
      t : Fin d → Nat
      ht : Membership.mem (d.finMulAntidiag n) t
      f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
      hf : ∀ (p : Nat) (hp : Membership.mem n.primeFactors p), (fun i => Dvd.dvd p ( …
      hf_unique : ∀ (p : Nat) (hp : Membership.mem n.primeFactors p) (y : Fin d), (f …
      i : Fin d
      this : Dvd.dvd (t i) n
      p : Nat
      hp : Membership.mem n.primeFactors p
      ⊢ Iff (Eq (f ↑⟨p, hp⟩ ⋯) i) (Dvd.dvd (↑⟨p, hp⟩) (t i))
    -/
    refine ⟨by rintro rfl; apply hf, fun h => (hf_unique p hp i h).symm⟩
    /-
      🎉 no goals
    -/
  /-
    d n : Nat
    hn : Squarefree n
    t : Fin d → Nat
    ht : Membership.mem (d.finMulAntidiag n) t
    f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
    hf : ∀ (p : Nat) (hp : Membership.mem n.primeFactors p), (fun i => Dvd.dvd p ( …
    hf_unique : ∀ (p : Nat) (hp : Membership.mem n.primeFactors p) (y : Fin d), (f …
    i : Fin d
    this : Dvd.dvd (t i) n
    ⊢ Eq (n.primeFactors.attach.prod fun p => ite (Dvd.dvd (↑p) (t i)) (↑p) 1) (t i)
  -/
  rw [prod_attach (f:=fun p => if p ∣ t i then p else 1), ← Finset.prod_filter]
  /-
    d n : Nat
    hn : Squarefree n
    t : Fin d → Nat
    ht : Membership.mem (d.finMulAntidiag n) t
    f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
    hf : ∀ (p : Nat) (hp : Membership.mem n.primeFactors p), (fun i => Dvd.dvd p ( …
    hf_unique : ∀ (p : Nat) (hp : Membership.mem n.primeFactors p) (y : Fin d), (f …
    i : Fin d
    this : Dvd.dvd (t i) n
    ⊢ Eq ((Finset.filter (fun a => Dvd.dvd a (t i)) n.primeFactors).prod fun a =>  …
  -/
  rw [primeFactors_filter_dvd_of_dvd hn.ne_zero this]
  /-
    d n : Nat
    hn : Squarefree n
    t : Fin d → Nat
    ht : Membership.mem (d.finMulAntidiag n) t
    f : (p : Nat) → Membership.mem n.primeFactors p → Fin d
    hf : ∀ (p : Nat) (hp : Membership.mem n.primeFactors p), (fun i => Dvd.dvd p ( …
    hf_unique : ∀ (p : Nat) (hp : Membership.mem n.primeFactors p) (y : Fin d), (f …
    i : Fin d
    this : Dvd.dvd (t i) n
    ⊢ Eq ((t i).primeFactors.prod fun a => a) (t i)
  -/
  exact prod_primeFactors_of_squarefree <| hn.squarefree_of_dvd this
  /-
    🎉 no goals
  -/


private theorem card_finMulAntidiag_pi (d n : ℕ) (hn : Squarefree n) :
    #(n.primeFactors.pi fun _ => (univ : Finset <| Fin d)) =
      #(finMulAntidiag d n) := by
  apply Finset.card_bij (Nat.primeFactorsPiBij d n) (primeFactorsPiBij_img d n hn)
    (primeFactorsPiBij_inj d n) (primeFactorsPiBij_surj d n hn)


theorem card_finMulAntidiag_of_squarefree {d n : ℕ} (hn : Squarefree n) :
    #(finMulAntidiag d n) = d ^ ω n := by
  rw [← card_finMulAntidiag_pi d n hn, Finset.card_pi, Finset.prod_const,
    ArithmeticFunction.cardDistinctFactors_apply, ← List.card_toFinset, toFinset_factors,
    Finset.card_fin]


theorem finMulAntidiag_three {n : ℕ} (a) (ha : a ∈ finMulAntidiag 3 n) : a 0 * a 1 * a 2 = n := by
  /-
    n : Nat
    a : Fin 3 → Nat
    ha : Membership.mem (Nat.finMulAntidiag 3 n) a
    ⊢ Eq (HMul.hMul (HMul.hMul (a 0) (a 1)) (a 2)) n
  -/
  rw [← (mem_finMulAntidiag.mp ha).1, Fin.prod_univ_three a]
  /-
    🎉 no goals
  -/


@[reducible]
private def f {n : ℕ} : ∀ a ∈ finMulAntidiag 3 n, ℕ × ℕ := fun a _ => (a 0 * a 1, a 0 * a 2)


private theorem f_img {n : ℕ} (hn : Squarefree n) (a : Fin 3 → ℕ)
    (ha : a ∈ finMulAntidiag 3 n) :
    f a ha ∈ Finset.filter (fun ⟨x, y⟩ => x.lcm y = n) (n.divisors ×ˢ n.divisors) := by
  /-
    n : Nat
    hn : Squarefree n
    a : Fin 3 → Nat
    ha : Membership.mem (Nat.finMulAntidiag 3 n) a
    ⊢ Membership.mem (Finset.filter (fun x => Nat.card_pair_lcm_eq.f_img.match_1 ( …
  -/
  rw [mem_filter, Finset.mem_product, mem_divisors, mem_divisors]
  /-
    n : Nat
    hn : Squarefree n
    a : Fin 3 → Nat
    ha : Membership.mem (Nat.finMulAntidiag 3 n) a
    ⊢ And (And (And (Dvd.dvd (Nat.card_pair_lcm_eq.f a ha).1 n) (Ne n 0)) (And (Dv …
  -/
  refine ⟨⟨⟨?_, hn.ne_zero⟩, ⟨?_, hn.ne_zero⟩⟩, ?_⟩ <;> rw [f, ← finMulAntidiag_three a ha]
    /-
      case refine_1
      n : Nat
      hn : Squarefree n
      a : Fin 3 → Nat
      ha : Membership.mem (Nat.finMulAntidiag 3 n) a
      ⊢ Dvd.dvd { fst := HMul.hMul (a 0) (a 1), snd := HMul.hMul (a 0) (a 2) }.1 (HM …
    -/
  · apply dvd_mul_right
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      n : Nat
      hn : Squarefree n
      a : Fin 3 → Nat
      ha : Membership.mem (Nat.finMulAntidiag 3 n) a
      ⊢ Dvd.dvd { fst := HMul.hMul (a 0) (a 1), snd := HMul.hMul (a 0) (a 2) }.2 (HM …
    -/
  · use a 1; ring
             /-
               🎉 no goals
             -/
  /-
    case refine_3
    n : Nat
    hn : Squarefree n
    a : Fin 3 → Nat
    ha : Membership.mem (Nat.finMulAntidiag 3 n) a
    ⊢ Nat.card_pair_lcm_eq.f_img.match_1 (fun x => Prop) { fst := HMul.hMul (a 0)  …
  -/
  dsimp only
  /-
    case refine_3
    n : Nat
    hn : Squarefree n
    a : Fin 3 → Nat
    ha : Membership.mem (Nat.finMulAntidiag 3 n) a
    ⊢ Eq ((HMul.hMul (a 0) (a 1)).lcm (HMul.hMul (a 0) (a 2))) (HMul.hMul (HMul.hM …
  -/
  rw [lcm_mul_left, Nat.Coprime.lcm_eq_mul]
    /-
      case refine_3
      n : Nat
      hn : Squarefree n
      a : Fin 3 → Nat
      ha : Membership.mem (Nat.finMulAntidiag 3 n) a
      ⊢ Eq (HMul.hMul (a 0) (HMul.hMul (a 1) (a 2))) (HMul.hMul (HMul.hMul (a 0) (a  …
    -/
  · ring
    /-
      🎉 no goals
    -/
  /-
    case refine_3
    n : Nat
    hn : Squarefree n
    a : Fin 3 → Nat
    ha : Membership.mem (Nat.finMulAntidiag 3 n) a
    ⊢ (a 1).Coprime (a 2)
  -/
  refine coprime_of_squarefree_mul (hn.squarefree_of_dvd ?_)
  /-
    case refine_3
    n : Nat
    hn : Squarefree n
    a : Fin 3 → Nat
    ha : Membership.mem (Nat.finMulAntidiag 3 n) a
    ⊢ Dvd.dvd (HMul.hMul (a 1) (a 2)) n
  -/
  use a 0; rw [← finMulAntidiag_three a ha]; ring
                                             /-
                                               🎉 no goals
                                             -/


private theorem f_inj {n : ℕ} (a : Fin 3 → ℕ) (ha : a ∈ finMulAntidiag 3 n)
    (b : Fin 3 → ℕ) (hb : b ∈ finMulAntidiag 3 n) (hfab : f a ha = f b hb) :
    a = b := by
  /-
    n : Nat
    a : Fin 3 → Nat
    ha : Membership.mem (Nat.finMulAntidiag 3 n) a
    b : Fin 3 → Nat
    hb : Membership.mem (Nat.finMulAntidiag 3 n) b
    hfab : Eq (Nat.card_pair_lcm_eq.f a ha) (Nat.card_pair_lcm_eq.f b hb)
    ⊢ Eq a b
  -/
  obtain ⟨hfab1, hfab2⟩ := Prod.mk.inj hfab
  have hprods : a 0 * a 1 * a 2 = a 0 * a 1 * b 2 := by
    rw [finMulAntidiag_three a ha, hfab1, finMulAntidiag_three b hb]
  have hab2 : a 2 = b 2 := by
    rw [← mul_right_inj' <| mul_ne_zero (ne_zero_of_mem_finMulAntidiag ha 0)
      (ne_zero_of_mem_finMulAntidiag ha 1)]
    exact hprods
  have hab0 : a 0 = b 0 := by
    rw [hab2] at hfab2
    exact (mul_left_inj' <| ne_zero_of_mem_finMulAntidiag hb 2).mp hfab2;
  have hab1 : a 1 = b 1 := by
    rw [hab0] at hfab1
    exact (mul_right_inj' <| ne_zero_of_mem_finMulAntidiag hb 0).mp hfab1;
  /-
    case intro
    n : Nat
    a : Fin 3 → Nat
    ha : Membership.mem (Nat.finMulAntidiag 3 n) a
    b : Fin 3 → Nat
    hb : Membership.mem (Nat.finMulAntidiag 3 n) b
    hfab : Eq (Nat.card_pair_lcm_eq.f a ha) (Nat.card_pair_lcm_eq.f b hb)
    hfab1 : Eq (HMul.hMul (a 0) (a 1)) (HMul.hMul (b 0) (b 1))
    hfab2 : Eq (HMul.hMul (a 0) (a 2)) (HMul.hMul (b 0) (b 2))
    hprods : Eq (HMul.hMul (HMul.hMul (a 0) (a 1)) (a 2)) (HMul.hMul (HMul.hMul (a …
    hab2 : Eq (a 2) (b 2)
    hab0 : Eq (a 0) (b 0)
    hab1 : Eq (a 1) (b 1)
    ⊢ Eq a b
  -/
                            /-
                              🎉 no goals
                            -/
                            /-
                              🎉 no goals
                            -/
  funext i; fin_cases i <;> assumption
                            /-
                              🎉 no goals
                            -/


private theorem f_surj {n : ℕ} (hn : n ≠ 0) (b : ℕ × ℕ)
    (hb : b ∈ Finset.filter (fun ⟨x, y⟩ => x.lcm y = n) (n.divisors ×ˢ n.divisors)) :
    ∃ (a : Fin 3 → ℕ) (ha : a ∈ finMulAntidiag 3 n), f a ha = b := by
  /-
    n : Nat
    hn : Ne n 0
    b : Prod Nat Nat
    hb : Membership.mem (Finset.filter (fun x => Nat.card_pair_lcm_eq.f_img.match_ …
    ⊢ Exists fun a => Exists fun ha => Eq (Nat.card_pair_lcm_eq.f a ha) b
  -/
  dsimp only at hb
  /-
    n : Nat
    hn : Ne n 0
    b : Prod Nat Nat
    hb : Membership.mem (Finset.filter (fun x => Eq (x.1.lcm x.2) n) (SProd.sprod  …
    ⊢ Exists fun a => Exists fun ha => Eq (Nat.card_pair_lcm_eq.f a ha) b
  -/
  let g := b.fst.gcd b.snd
  /-
    n : Nat
    hn : Ne n 0
    b : Prod Nat Nat
    hb : Membership.mem (Finset.filter (fun x => Eq (x.1.lcm x.2) n) (SProd.sprod  …
    g : Nat := b.1.gcd b.2
    ⊢ Exists fun a => Exists fun ha => Eq (Nat.card_pair_lcm_eq.f a ha) b
  -/
  let a := ![g, b.fst/g, b.snd/g]
  have ha : a ∈ finMulAntidiag 3 n := by
    rw [mem_finMulAntidiag]
    rw [mem_filter, Finset.mem_product] at hb
    refine ⟨?_, hn⟩
    · rw [Fin.prod_univ_three a]
      simp only [a, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
      Matrix.cons_val_two, Matrix.tail_cons]
      rw [Nat.mul_div_cancel_left' (Nat.gcd_dvd_left _ _), ← hb.2, lcm,
        Nat.mul_div_assoc b.fst (Nat.gcd_dvd_right b.fst b.snd)]
  /-
    n : Nat
    hn : Ne n 0
    b : Prod Nat Nat
    hb : Membership.mem (Finset.filter (fun x => Eq (x.1.lcm x.2) n) (SProd.sprod  …
    g : Nat := b.1.gcd b.2
    a : Fin (Nat.succ 0).succ.succ → Nat := Matrix.vecCons g (Matrix.vecCons (HDiv …
    ha : Membership.mem (Nat.finMulAntidiag 3 n) a
    ⊢ Exists fun a => Exists fun ha => Eq (Nat.card_pair_lcm_eq.f a ha) b
  -/
  use a; use ha
  /-
    case h
    n : Nat
    hn : Ne n 0
    b : Prod Nat Nat
    hb : Membership.mem (Finset.filter (fun x => Eq (x.1.lcm x.2) n) (SProd.sprod  …
    g : Nat := b.1.gcd b.2
    a : Fin (Nat.succ 0).succ.succ → Nat := Matrix.vecCons g (Matrix.vecCons (HDiv …
    ha : Membership.mem (Nat.finMulAntidiag 3 n) a
    ⊢ Eq (Nat.card_pair_lcm_eq.f a ha) b
  -/
  apply Prod.ext <;> simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]
        /-
          case h.fst
          n : Nat
          hn : Ne n 0
          b : Prod Nat Nat
          hb : Membership.mem (Finset.filter (fun x => Eq (x.1.lcm x.2) n) (SProd.sprod  …
          g : Nat := b.1.gcd b.2
          a : Fin (Nat.succ 0).succ.succ → Nat := Matrix.vecCons g (Matrix.vecCons (HDiv …
          ha : Membership.mem (Nat.finMulAntidiag 3 n) a
          ⊢ Eq (HMul.hMul (a 0) (a 1)) b.1
        -/
    <;> apply Nat.mul_div_cancel'
    /-
      case h.fst.H
      n : Nat
      hn : Ne n 0
      b : Prod Nat Nat
      hb : Membership.mem (Finset.filter (fun x => Eq (x.1.lcm x.2) n) (SProd.sprod  …
      g : Nat := b.1.gcd b.2
      a : Fin (Nat.succ 0).succ.succ → Nat := Matrix.vecCons g (Matrix.vecCons (HDiv …
      ha : Membership.mem (Nat.finMulAntidiag 3 n) a
      ⊢ Dvd.dvd (a 0) b.1
    -/
  · apply Nat.gcd_dvd_left
    /-
      🎉 no goals
    -/
    /-
      case h.snd.H
      n : Nat
      hn : Ne n 0
      b : Prod Nat Nat
      hb : Membership.mem (Finset.filter (fun x => Eq (x.1.lcm x.2) n) (SProd.sprod  …
      g : Nat := b.1.gcd b.2
      a : Fin (Nat.succ 0).succ.succ → Nat := Matrix.vecCons g (Matrix.vecCons (HDiv …
      ha : Membership.mem (Nat.finMulAntidiag 3 n) a
      ⊢ Dvd.dvd (a 0) b.2
    -/
  · apply Nat.gcd_dvd_right
    /-
      🎉 no goals
    -/


open card_pair_lcm_eq in
theorem card_pair_lcm_eq {n : ℕ} (hn : Squarefree n) :
    #{p ∈ (n.divisors ×ˢ n.divisors) | p.1.lcm p.2 = n} = 3 ^ ω n := by
  /-
    n : Nat
    hn : Squarefree n
    ⊢ Eq (Finset.filter (fun p => Eq (p.1.lcm p.2) n) (SProd.sprod n.divisors n.di …
  -/
  rw [← card_finMulAntidiag_of_squarefree hn, eq_comm]
  /-
    n : Nat
    hn : Squarefree n
    ⊢ Eq (Nat.finMulAntidiag 3 n).card (Finset.filter (fun p => Eq (p.1.lcm p.2) n …
  -/
  apply Finset.card_bij f (f_img hn) (f_inj) (f_surj hn.ne_zero)
  /-
    🎉 no goals
  -/


