theorem mem_span_C_X_sub_C_X_sub_C_iff_eval_eval_eq_zero {b : R[X]} {P : R[X][X]} :
    P ∈ Ideal.span {C (X - C a), X - C b} ↔ (P.eval b).eval a = 0 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    a : R
    b : Polynomial R
    P : Polynomial (Polynomial R)
    ⊢ Iff (Membership.mem (Ideal.span (Insert.insert (Polynomial.C (HSub.hSub Poly …
  -/
  rw [Ideal.mem_span_pair]
  /-
    R : Type u_1
    inst✝ : CommRing R
    a : R
    b : Polynomial R
    P : Polynomial (Polynomial R)
    ⊢ Iff (Exists fun a_1 => Exists fun b_1 => Eq (HAdd.hAdd (HMul.hMul a_1 (Polyn …
  -/
  constructor <;> intro h
    /-
      case mp
      R : Type u_1
      inst✝ : CommRing R
      a : R
      b : Polynomial R
      P : Polynomial (Polynomial R)
      h : Exists fun a_1 => Exists fun b_1 => Eq (HAdd.hAdd (HMul.hMul a_1 (Polynomi …
      ⊢ Eq (Polynomial.eval a (Polynomial.eval b P)) 0
    -/
  · rcases h with ⟨_, _, rfl⟩
    /-
      case mp.intro.intro
      R : Type u_1
      inst✝ : CommRing R
      a : R
      b : Polynomial R
      w✝¹ w✝ : Polynomial (Polynomial R)
      ⊢ Eq (Polynomial.eval a (Polynomial.eval b (HAdd.hAdd (HMul.hMul w✝¹ (Polynomi …
    -/
    simp only [eval_C, eval_X, eval_add, eval_sub, eval_mul, add_zero, mul_zero, sub_self]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝ : CommRing R
      a : R
      b : Polynomial R
      P : Polynomial (Polynomial R)
      h : Eq (Polynomial.eval a (Polynomial.eval b P)) 0
      ⊢ Exists fun a_1 => Exists fun b_1 => Eq (HAdd.hAdd (HMul.hMul a_1 (Polynomial …
    -/
  · rcases dvd_iff_isRoot.mpr h with ⟨p, hp⟩
    /-
      case mpr.intro
      R : Type u_1
      inst✝ : CommRing R
      a : R
      b : Polynomial R
      P : Polynomial (Polynomial R)
      h : Eq (Polynomial.eval a (Polynomial.eval b P)) 0
      p : Polynomial R
      hp : Eq (Polynomial.eval b P) (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C …
      ⊢ Exists fun a_1 => Exists fun b_1 => Eq (HAdd.hAdd (HMul.hMul a_1 (Polynomial …
    -/
    rcases @X_sub_C_dvd_sub_C_eval _ b _ P with ⟨q, hq⟩
    /-
      case mpr.intro.intro
      R : Type u_1
      inst✝ : CommRing R
      a : R
      b : Polynomial R
      P : Polynomial (Polynomial R)
      h : Eq (Polynomial.eval a (Polynomial.eval b P)) 0
      p : Polynomial R
      hp : Eq (Polynomial.eval b P) (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C …
      q : Polynomial (Polynomial R)
      hq : Eq (HSub.hSub P (Polynomial.C (Polynomial.eval b P))) (HMul.hMul (HSub.hS …
      ⊢ Exists fun a_1 => Exists fun b_1 => Eq (HAdd.hAdd (HMul.hMul a_1 (Polynomial …
    -/
    exact ⟨C p, q, by rw [mul_comm, mul_comm q, eq_add_of_sub_eq' hq, hp, C_mul]⟩
    /-
      🎉 no goals
    -/


theorem ker_evalRingHom (x : R) : RingHom.ker (evalRingHom x) = Ideal.span {X - C x} := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    x : R
    ⊢ Eq (RingHom.ker (Polynomial.evalRingHom x)) (Ideal.span (Singleton.singleton …
  -/
  ext y
  /-
    case h
    R : Type u_1
    inst✝ : CommRing R
    x : R
    y : Polynomial R
    ⊢ Iff (Membership.mem (RingHom.ker (Polynomial.evalRingHom x)) y) (Membership. …
  -/
  simp [Ideal.mem_span_singleton, dvd_iff_isRoot, RingHom.mem_ker]
  /-
    🎉 no goals
  -/


@[simp]
theorem ker_modByMonicHom {q : R[X]} (hq : q.Monic) :
    LinearMap.ker (Polynomial.modByMonicHom q) = (Ideal.span {q}).restrictScalars R :=
  Submodule.ext fun _ => (mem_ker_modByMonic hq).trans Ideal.mem_span_singleton.symm


open Algebra in
lemma _root_.Algebra.mem_ideal_map_adjoin {R S : Type*} [CommRing R] [CommRing S] [Algebra R S]
    (x : S) (I : Ideal R) {y : adjoin R ({x} : Set S)} :
    y ∈ I.map (algebraMap R (adjoin R ({x} : Set S))) ↔
      ∃ p : R[X], (∀ i, p.coeff i ∈ I) ∧ Polynomial.aeval x p = y := by
  /-
    R : Type u_2
    S : Type u_3
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    x : S
    I : Ideal R
    y : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.singleton x …
    ⊢ Iff (Membership.mem (Ideal.map (algebraMap R (Subtype fun x_1 => Membership. …
  -/
  constructor
    /-
      case mp
      R : Type u_2
      S : Type u_3
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x : S
      I : Ideal R
      y : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.singleton x …
      ⊢ Membership.mem (Ideal.map (algebraMap R (Subtype fun x_1 => Membership.mem ( …
    -/
  · intro H
    /-
      case mp
      R : Type u_2
      S : Type u_3
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x : S
      I : Ideal R
      y : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.singleton x …
      H : Membership.mem (Ideal.map (algebraMap R (Subtype fun x_1 => Membership.mem …
      ⊢ Exists fun p => And (∀ (i : Nat), Membership.mem I (p.coeff i)) (Eq ((Polyno …
    -/
    induction' H using Submodule.span_induction with a ha a b ha hb ha' hb' a b hb hb'
      /-
        case mp.mem
        R : Type u_2
        S : Type u_3
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        x : S
        I : Ideal R
        y a : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.singleton …
        ha : Membership.mem (Set.image ⇑(algebraMap R (Subtype fun x_1 => Membership.m …
        ⊢ Exists fun p => And (∀ (i : Nat), Membership.mem I (p.coeff i)) (Eq ((Polyno …
      -/
    · obtain ⟨a, ha, rfl⟩ := ha
      /-
        case mp.mem.intro.intro
        R : Type u_2
        S : Type u_3
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        x : S
        I : Ideal R
        y : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.singleton x …
        a : R
        ha : Membership.mem (↑I) a
        ⊢ Exists fun p => And (∀ (i : Nat), Membership.mem I (p.coeff i)) (Eq ((Polyno …
      -/
      exact ⟨C a, fun i ↦ by rw [coeff_C]; aesop, aeval_C _ _⟩
      /-
        🎉 no goals
      -/
      /-
        case mp.zero
        R : Type u_2
        S : Type u_3
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        x : S
        I : Ideal R
        y : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.singleton x …
        ⊢ Exists fun p => And (∀ (i : Nat), Membership.mem I (p.coeff i)) (Eq ((Polyno …
      -/
    · exact ⟨0, by simp, aeval_zero _⟩
      /-
        🎉 no goals
      -/
      /-
        case mp.add
        R : Type u_2
        S : Type u_3
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        x : S
        I : Ideal R
        y a b : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.singlet …
        ha : Membership.mem (Submodule.span (Subtype fun x_1 => Membership.mem (Algebr …
        hb : Membership.mem (Submodule.span (Subtype fun x_1 => Membership.mem (Algebr …
        ha' : Exists fun p => And (∀ (i : Nat), Membership.mem I (p.coeff i)) (Eq ((Po …
        hb' : Exists fun p => And (∀ (i : Nat), Membership.mem I (p.coeff i)) (Eq ((Po …
        ⊢ Exists fun p => And (∀ (i : Nat), Membership.mem I (p.coeff i)) (Eq ((Polyno …
      -/
    · obtain ⟨a, ha, ha'⟩ := ha'
      /-
        case mp.add.intro.intro
        R : Type u_2
        S : Type u_3
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        x : S
        I : Ideal R
        y a✝ b : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.single …
        ha✝ : Membership.mem (Submodule.span (Subtype fun x_1 => Membership.mem (Algeb …
        hb : Membership.mem (Submodule.span (Subtype fun x_1 => Membership.mem (Algebr …
        hb' : Exists fun p => And (∀ (i : Nat), Membership.mem I (p.coeff i)) (Eq ((Po …
        a : Polynomial R
        ha : ∀ (i : Nat), Membership.mem I (a.coeff i)
        ha' : Eq ((Polynomial.aeval x) a) ↑a✝
        ⊢ Exists fun p => And (∀ (i : Nat), Membership.mem I (p.coeff i)) (Eq ((Polyno …
      -/
      obtain ⟨b, hb, hb'⟩ := hb'
      /-
        case mp.add.intro.intro.intro.intro
        R : Type u_2
        S : Type u_3
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        x : S
        I : Ideal R
        y a✝ b✝ : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.singl …
        ha✝ : Membership.mem (Submodule.span (Subtype fun x_1 => Membership.mem (Algeb …
        hb✝ : Membership.mem (Submodule.span (Subtype fun x_1 => Membership.mem (Algeb …
        a : Polynomial R
        ha : ∀ (i : Nat), Membership.mem I (a.coeff i)
        ha' : Eq ((Polynomial.aeval x) a) ↑a✝
        b : Polynomial R
        hb : ∀ (i : Nat), Membership.mem I (b.coeff i)
        hb' : Eq ((Polynomial.aeval x) b) ↑b✝
        ⊢ Exists fun p => And (∀ (i : Nat), Membership.mem I (p.coeff i)) (Eq ((Polyno …
      -/
      exact ⟨a + b, fun i ↦ by simpa using add_mem (ha i) (hb i), by simp [ha', hb']⟩
      /-
        🎉 no goals
      -/
      /-
        case mp.smul
        R : Type u_2
        S : Type u_3
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        x : S
        I : Ideal R
        y a b : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.singlet …
        hb : Membership.mem (Submodule.span (Subtype fun x_1 => Membership.mem (Algebr …
        hb' : Exists fun p => And (∀ (i : Nat), Membership.mem I (p.coeff i)) (Eq ((Po …
        ⊢ Exists fun p => And (∀ (i : Nat), Membership.mem I (p.coeff i)) (Eq ((Polyno …
      -/
    · obtain ⟨b', hb, hb'⟩ := hb'
      /-
        case mp.smul.intro.intro
        R : Type u_2
        S : Type u_3
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        x : S
        I : Ideal R
        y a b : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.singlet …
        hb✝ : Membership.mem (Submodule.span (Subtype fun x_1 => Membership.mem (Algeb …
        b' : Polynomial R
        hb : ∀ (i : Nat), Membership.mem I (b'.coeff i)
        hb' : Eq ((Polynomial.aeval x) b') ↑b
        ⊢ Exists fun p => And (∀ (i : Nat), Membership.mem I (p.coeff i)) (Eq ((Polyno …
      -/
      obtain ⟨a, ha⟩ := a
      /-
        case mp.smul.intro.intro.mk
        R : Type u_2
        S : Type u_3
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        x : S
        I : Ideal R
        y b : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.singleton …
        hb✝ : Membership.mem (Submodule.span (Subtype fun x_1 => Membership.mem (Algeb …
        b' : Polynomial R
        hb : ∀ (i : Nat), Membership.mem I (b'.coeff i)
        hb' : Eq ((Polynomial.aeval x) b') ↑b
        a : S
        ha : Membership.mem (Algebra.adjoin R (Singleton.singleton x)) a
        ⊢ Exists fun p => And (∀ (i : Nat), Membership.mem I (p.coeff i)) (Eq ((Polyno …
      -/
      rw [Algebra.adjoin_singleton_eq_range_aeval] at ha
      /-
        case mp.smul.intro.intro.mk
        R : Type u_2
        S : Type u_3
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        x : S
        I : Ideal R
        y b : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.singleton …
        hb✝ : Membership.mem (Submodule.span (Subtype fun x_1 => Membership.mem (Algeb …
        b' : Polynomial R
        hb : ∀ (i : Nat), Membership.mem I (b'.coeff i)
        hb' : Eq ((Polynomial.aeval x) b') ↑b
        a : S
        ha✝ : Membership.mem (Algebra.adjoin R (Singleton.singleton x)) a
        ha : Membership.mem (Polynomial.aeval x).range a
        ⊢ Exists fun p => And (∀ (i : Nat), Membership.mem I (p.coeff i)) (Eq ((Polyno …
      -/
      obtain ⟨p, hp : aeval x p = a⟩ := ha
      /-
        case mp.smul.intro.intro.mk.intro
        R : Type u_2
        S : Type u_3
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        x : S
        I : Ideal R
        y b : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.singleton …
        hb✝ : Membership.mem (Submodule.span (Subtype fun x_1 => Membership.mem (Algeb …
        b' : Polynomial R
        hb : ∀ (i : Nat), Membership.mem I (b'.coeff i)
        hb' : Eq ((Polynomial.aeval x) b') ↑b
        a : S
        ha : Membership.mem (Algebra.adjoin R (Singleton.singleton x)) a
        p : Polynomial R
        hp : Eq ((Polynomial.aeval x) p) a
        ⊢ Exists fun p => And (∀ (i : Nat), Membership.mem I (p.coeff i)) (Eq ((Polyno …
      -/
      refine ⟨p * b', fun i ↦ ?_, by simp [hp, hb']⟩
      /-
        case mp.smul.intro.intro.mk.intro
        R : Type u_2
        S : Type u_3
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        x : S
        I : Ideal R
        y b : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.singleton …
        hb✝ : Membership.mem (Submodule.span (Subtype fun x_1 => Membership.mem (Algeb …
        b' : Polynomial R
        hb : ∀ (i : Nat), Membership.mem I (b'.coeff i)
        hb' : Eq ((Polynomial.aeval x) b') ↑b
        a : S
        ha : Membership.mem (Algebra.adjoin R (Singleton.singleton x)) a
        p : Polynomial R
        hp : Eq ((Polynomial.aeval x) p) a
        i : Nat
        ⊢ Membership.mem I ((HMul.hMul p b').coeff i)
      -/
      rw [coeff_mul]
      /-
        case mp.smul.intro.intro.mk.intro
        R : Type u_2
        S : Type u_3
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        x : S
        I : Ideal R
        y b : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.singleton …
        hb✝ : Membership.mem (Submodule.span (Subtype fun x_1 => Membership.mem (Algeb …
        b' : Polynomial R
        hb : ∀ (i : Nat), Membership.mem I (b'.coeff i)
        hb' : Eq ((Polynomial.aeval x) b') ↑b
        a : S
        ha : Membership.mem (Algebra.adjoin R (Singleton.singleton x)) a
        p : Polynomial R
        hp : Eq ((Polynomial.aeval x) p) a
        i : Nat
        ⊢ Membership.mem I ((Finset.HasAntidiagonal.antidiagonal i).sum fun x => HMul. …
      -/
      exact sum_mem fun i hi ↦ Ideal.mul_mem_left _ _ (hb _)
      /-
        🎉 no goals
      -/
    /-
      case mpr
      R : Type u_2
      S : Type u_3
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x : S
      I : Ideal R
      y : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.singleton x …
      ⊢ (Exists fun p => And (∀ (i : Nat), Membership.mem I (p.coeff i)) (Eq ((Polyn …
    -/
  · rintro ⟨p, hp, hp'⟩
    have : y = ∑ i in p.support, p.coeff i • ⟨_, (X ^ i).aeval_mem_adjoin_singleton _ x⟩ := by
      trans ∑ i in p.support, ⟨_, (C (p.coeff i) * X ^ i).aeval_mem_adjoin_singleton _ x⟩
      · ext1
        simp only [AddSubmonoidClass.coe_finset_sum, ← map_sum, ← hp', ← as_sum_support_C_mul_X_pow]
      · congr with i
        simp [Algebra.smul_def]
    /-
      case mpr.intro.intro
      R : Type u_2
      S : Type u_3
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x : S
      I : Ideal R
      y : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.singleton x …
      p : Polynomial R
      hp : ∀ (i : Nat), Membership.mem I (p.coeff i)
      hp' : Eq ((Polynomial.aeval x) p) ↑y
      this : Eq y (p.support.sum fun i => HSMul.hSMul (p.coeff i) ⟨(Polynomial.aeval …
      ⊢ Membership.mem (Ideal.map (algebraMap R (Subtype fun x_1 => Membership.mem ( …
    -/
    simp_rw [this, Algebra.smul_def]
    /-
      case mpr.intro.intro
      R : Type u_2
      S : Type u_3
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x : S
      I : Ideal R
      y : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.singleton x …
      p : Polynomial R
      hp : ∀ (i : Nat), Membership.mem I (p.coeff i)
      hp' : Eq ((Polynomial.aeval x) p) ↑y
      this : Eq y (p.support.sum fun i => HSMul.hSMul (p.coeff i) ⟨(Polynomial.aeval …
      ⊢ Membership.mem (Ideal.map (algebraMap R (Subtype fun x_1 => Membership.mem ( …
    -/
    exact sum_mem fun i _ ↦ Ideal.mul_mem_right _ _ (Ideal.mem_map_of_mem _ (hp i))
    /-
      🎉 no goals
    -/


