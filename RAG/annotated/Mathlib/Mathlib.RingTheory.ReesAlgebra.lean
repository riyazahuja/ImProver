/-- The Rees algebra of an ideal `I`, defined as the subalgebra of `R[X]` whose `i`-th coefficient
falls in `I ^ i`. -/
def reesAlgebra : Subalgebra R R[X] where
  carrier := { f | ∀ i, f.coeff i ∈ I ^ i }
  mul_mem' hf hg i := by
    /-
      R M : Type u
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      a✝ b✝ : Polynomial R
      hf : Membership.mem (setOf fun f => ∀ (i : Nat), Membership.mem (HPow.hPow I i …
      hg : Membership.mem (setOf fun f => ∀ (i : Nat), Membership.mem (HPow.hPow I i …
      i : Nat
      ⊢ Membership.mem (HPow.hPow I i) ((HMul.hMul a✝ b✝).coeff i)
    -/
    rw [coeff_mul]
    /-
      R M : Type u
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      a✝ b✝ : Polynomial R
      hf : Membership.mem (setOf fun f => ∀ (i : Nat), Membership.mem (HPow.hPow I i …
      hg : Membership.mem (setOf fun f => ∀ (i : Nat), Membership.mem (HPow.hPow I i …
      i : Nat
      ⊢ Membership.mem (HPow.hPow I i) ((Finset.HasAntidiagonal.antidiagonal i).sum  …
    -/
    apply Ideal.sum_mem
    /-
      case a
      R M : Type u
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      a✝ b✝ : Polynomial R
      hf : Membership.mem (setOf fun f => ∀ (i : Nat), Membership.mem (HPow.hPow I i …
      hg : Membership.mem (setOf fun f => ∀ (i : Nat), Membership.mem (HPow.hPow I i …
      i : Nat
      ⊢ ∀ (c : Prod Nat Nat), Membership.mem (Finset.HasAntidiagonal.antidiagonal i) …
    -/
    rintro ⟨j, k⟩ e
    /-
      case a.mk
      R M : Type u
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      a✝ b✝ : Polynomial R
      hf : Membership.mem (setOf fun f => ∀ (i : Nat), Membership.mem (HPow.hPow I i …
      hg : Membership.mem (setOf fun f => ∀ (i : Nat), Membership.mem (HPow.hPow I i …
      i j k : Nat
      e : Membership.mem (Finset.HasAntidiagonal.antidiagonal i) { fst := j, snd :=  …
      ⊢ Membership.mem (HPow.hPow I i) (HMul.hMul (a✝.coeff { fst := j, snd := k }.1 …
    -/
    rw [← Finset.mem_antidiagonal.mp e, pow_add]
    /-
      case a.mk
      R M : Type u
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      a✝ b✝ : Polynomial R
      hf : Membership.mem (setOf fun f => ∀ (i : Nat), Membership.mem (HPow.hPow I i …
      hg : Membership.mem (setOf fun f => ∀ (i : Nat), Membership.mem (HPow.hPow I i …
      i j k : Nat
      e : Membership.mem (Finset.HasAntidiagonal.antidiagonal i) { fst := j, snd :=  …
      ⊢ Membership.mem (HMul.hMul (HPow.hPow I { fst := j, snd := k }.1) (HPow.hPow  …
    -/
    exact Ideal.mul_mem_mul (hf j) (hg k)
    /-
      🎉 no goals
    -/
  one_mem' i := by
    /-
      R M : Type u
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      i : Nat
      ⊢ Membership.mem (HPow.hPow I i) (Polynomial.coeff 1 i)
    -/
    rw [coeff_one]
    /-
      R M : Type u
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      i : Nat
      ⊢ Membership.mem (HPow.hPow I i) (ite (Eq i 0) 1 0)
    -/
    split_ifs with h
      /-
        case pos
        R M : Type u
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        I : Ideal R
        i : Nat
        h : Eq i 0
        ⊢ Membership.mem (HPow.hPow I i) 1
      -/
    · subst h
      /-
        case pos
        R M : Type u
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        I : Ideal R
        ⊢ Membership.mem (HPow.hPow I 0) 1
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case neg
        R M : Type u
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        I : Ideal R
        i : Nat
        h : Not (Eq i 0)
        ⊢ Membership.mem (HPow.hPow I i) 0
      -/
    · simp
      /-
        🎉 no goals
      -/
  add_mem' hf hg i := by
    /-
      R M : Type u
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      a✝ b✝ : Polynomial R
      hf : Membership.mem { carrier := setOf fun f => ∀ (i : Nat), Membership.mem (H …
      hg : Membership.mem { carrier := setOf fun f => ∀ (i : Nat), Membership.mem (H …
      i : Nat
      ⊢ Membership.mem (HPow.hPow I i) ((HAdd.hAdd a✝ b✝).coeff i)
    -/
    rw [coeff_add]
    /-
      R M : Type u
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      a✝ b✝ : Polynomial R
      hf : Membership.mem { carrier := setOf fun f => ∀ (i : Nat), Membership.mem (H …
      hg : Membership.mem { carrier := setOf fun f => ∀ (i : Nat), Membership.mem (H …
      i : Nat
      ⊢ Membership.mem (HPow.hPow I i) (HAdd.hAdd (a✝.coeff i) (b✝.coeff i))
    -/
    exact Ideal.add_mem _ (hf i) (hg i)
    /-
      🎉 no goals
    -/
  zero_mem' _ := Ideal.zero_mem _
  algebraMap_mem' r i := by
    /-
      R M : Type u
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      r : R
      i : Nat
      ⊢ Membership.mem (HPow.hPow I i) (((algebraMap R (Polynomial R)) r).coeff i)
    -/
    rw [algebraMap_apply, coeff_C]
    /-
      R M : Type u
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      r : R
      i : Nat
      ⊢ Membership.mem (HPow.hPow I i) (ite (Eq i 0) ((algebraMap R R) r) 0)
    -/
    split_ifs with h
      /-
        case pos
        R M : Type u
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        I : Ideal R
        r : R
        i : Nat
        h : Eq i 0
        ⊢ Membership.mem (HPow.hPow I i) ((algebraMap R R) r)
      -/
    · subst h
      /-
        case pos
        R M : Type u
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        I : Ideal R
        r : R
        ⊢ Membership.mem (HPow.hPow I 0) ((algebraMap R R) r)
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case neg
        R M : Type u
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        I : Ideal R
        r : R
        i : Nat
        h : Not (Eq i 0)
        ⊢ Membership.mem (HPow.hPow I i) 0
      -/
    · simp
      /-
        🎉 no goals
      -/


theorem mem_reesAlgebra_iff (f : R[X]) : f ∈ reesAlgebra I ↔ ∀ i, f.coeff i ∈ I ^ i :=
  Iff.rfl


theorem mem_reesAlgebra_iff_support (f : R[X]) :
    f ∈ reesAlgebra I ↔ ∀ i ∈ f.support, f.coeff i ∈ I ^ i := by
  /-
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    f : Polynomial R
    ⊢ Iff (Membership.mem (reesAlgebra I) f) (∀ (i : Nat), Membership.mem f.suppor …
  -/
  apply forall_congr'
  /-
    case h
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    f : Polynomial R
    ⊢ ∀ (a : Nat), Iff (Membership.mem (HPow.hPow I a) (f.coeff a)) (Membership.me …
  -/
  intro a
  /-
    case h
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    f : Polynomial R
    a : Nat
    ⊢ Iff (Membership.mem (HPow.hPow I a) (f.coeff a)) (Membership.mem f.support a …
  -/
  rw [mem_support_iff, Iff.comm, Classical.imp_iff_right_iff, Ne, ← imp_iff_not_or]
  /-
    case h
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    f : Polynomial R
    a : Nat
    ⊢ Eq (f.coeff a) 0 → Membership.mem (HPow.hPow I a) (f.coeff a)
  -/
  exact fun e => e.symm ▸ (I ^ a).zero_mem
  /-
    🎉 no goals
  -/


theorem reesAlgebra.monomial_mem {I : Ideal R} {i : ℕ} {r : R} :
    monomial i r ∈ reesAlgebra I ↔ r ∈ I ^ i := by
  simp +contextual [mem_reesAlgebra_iff_support, coeff_monomial, ←
    imp_iff_not_or]


theorem monomial_mem_adjoin_monomial {I : Ideal R} {n : ℕ} {r : R} (hr : r ∈ I ^ n) :
    monomial n r ∈ Algebra.adjoin R (Submodule.map (monomial 1 : R →ₗ[R] R[X]) I : Set R[X]) := by
  /-
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    n : Nat
    r : R
    hr : Membership.mem (HPow.hPow I n) r
    ⊢ Membership.mem (Algebra.adjoin R ↑(Submodule.map (Polynomial.monomial 1) I)) …
  -/
  induction' n with n hn generalizing r
    /-
      case zero
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      r : R
      hr : Membership.mem (HPow.hPow I 0) r
      ⊢ Membership.mem (Algebra.adjoin R ↑(Submodule.map (Polynomial.monomial 1) I)) …
    -/
  · exact Subalgebra.algebraMap_mem _ _
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      n : Nat
      hn : ∀ {r : R}, Membership.mem (HPow.hPow I n) r → Membership.mem (Algebra.adj …
      r : R
      hr : Membership.mem (HPow.hPow I (HAdd.hAdd n 1)) r
      ⊢ Membership.mem (Algebra.adjoin R ↑(Submodule.map (Polynomial.monomial 1) I)) …
    -/
  · rw [pow_succ'] at hr
    apply Submodule.smul_induction_on
      -- Porting note: did not need help with motive previously
      (p := fun r => (monomial (Nat.succ n)) r ∈ Algebra.adjoin R (Submodule.map (monomial 1) I)) hr
      /-
        case succ.smul
        R : Type u
        inst✝ : CommRing R
        I : Ideal R
        n : Nat
        hn : ∀ {r : R}, Membership.mem (HPow.hPow I n) r → Membership.mem (Algebra.adj …
        r : R
        hr : Membership.mem (HMul.hMul I (HPow.hPow I n)) r
        ⊢ ∀ (r : R), Membership.mem I r → ∀ (n_1 : R), Membership.mem (HPow.hPow I n)  …
      -/
    · intro r hr s hs
      /-
        case succ.smul
        R : Type u
        inst✝ : CommRing R
        I : Ideal R
        n : Nat
        hn : ∀ {r : R}, Membership.mem (HPow.hPow I n) r → Membership.mem (Algebra.adj …
        r✝ : R
        hr✝ : Membership.mem (HMul.hMul I (HPow.hPow I n)) r✝
        r : R
        hr : Membership.mem I r
        s : R
        hs : Membership.mem (HPow.hPow I n) s
        ⊢ Membership.mem (Algebra.adjoin R ↑(Submodule.map (Polynomial.monomial 1) I)) …
      -/
      rw [Nat.succ_eq_one_add, smul_eq_mul, ← monomial_mul_monomial]
      /-
        case succ.smul
        R : Type u
        inst✝ : CommRing R
        I : Ideal R
        n : Nat
        hn : ∀ {r : R}, Membership.mem (HPow.hPow I n) r → Membership.mem (Algebra.adj …
        r✝ : R
        hr✝ : Membership.mem (HMul.hMul I (HPow.hPow I n)) r✝
        r : R
        hr : Membership.mem I r
        s : R
        hs : Membership.mem (HPow.hPow I n) s
        ⊢ Membership.mem (Algebra.adjoin R ↑(Submodule.map (Polynomial.monomial 1) I)) …
      -/
      exact Subalgebra.mul_mem _ (Algebra.subset_adjoin (Set.mem_image_of_mem _ hr)) (hn hs)
      /-
        🎉 no goals
      -/
      /-
        case succ.add
        R : Type u
        inst✝ : CommRing R
        I : Ideal R
        n : Nat
        hn : ∀ {r : R}, Membership.mem (HPow.hPow I n) r → Membership.mem (Algebra.adj …
        r : R
        hr : Membership.mem (HMul.hMul I (HPow.hPow I n)) r
        ⊢ ∀ (x y : R), Membership.mem (Algebra.adjoin R ↑(Submodule.map (Polynomial.mo …
      -/
    · intro x y hx hy
      /-
        case succ.add
        R : Type u
        inst✝ : CommRing R
        I : Ideal R
        n : Nat
        hn : ∀ {r : R}, Membership.mem (HPow.hPow I n) r → Membership.mem (Algebra.adj …
        r : R
        hr : Membership.mem (HMul.hMul I (HPow.hPow I n)) r
        x y : R
        hx : Membership.mem (Algebra.adjoin R ↑(Submodule.map (Polynomial.monomial 1)  …
        hy : Membership.mem (Algebra.adjoin R ↑(Submodule.map (Polynomial.monomial 1)  …
        ⊢ Membership.mem (Algebra.adjoin R ↑(Submodule.map (Polynomial.monomial 1) I)) …
      -/
      rw [monomial_add]
      /-
        case succ.add
        R : Type u
        inst✝ : CommRing R
        I : Ideal R
        n : Nat
        hn : ∀ {r : R}, Membership.mem (HPow.hPow I n) r → Membership.mem (Algebra.adj …
        r : R
        hr : Membership.mem (HMul.hMul I (HPow.hPow I n)) r
        x y : R
        hx : Membership.mem (Algebra.adjoin R ↑(Submodule.map (Polynomial.monomial 1)  …
        hy : Membership.mem (Algebra.adjoin R ↑(Submodule.map (Polynomial.monomial 1)  …
        ⊢ Membership.mem (Algebra.adjoin R ↑(Submodule.map (Polynomial.monomial 1) I)) …
      -/
      exact Subalgebra.add_mem _ hx hy
      /-
        🎉 no goals
      -/


theorem adjoin_monomial_eq_reesAlgebra :
    Algebra.adjoin R (Submodule.map (monomial 1 : R →ₗ[R] R[X]) I : Set R[X]) = reesAlgebra I := by
  /-
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    ⊢ Eq (Algebra.adjoin R ↑(Submodule.map (Polynomial.monomial 1) I)) (reesAlgebr …
  -/
  apply le_antisymm
    /-
      case a
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      ⊢ LE.le (Algebra.adjoin R ↑(Submodule.map (Polynomial.monomial 1) I)) (reesAlg …
    -/
  · apply Algebra.adjoin_le _
    /-
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      ⊢ HasSubset.Subset ↑(Submodule.map (Polynomial.monomial 1) I) ↑(reesAlgebra I)
    -/
    rintro _ ⟨r, hr, rfl⟩
    /-
      case intro.intro
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      r : R
      hr : Membership.mem (↑I) r
      ⊢ Membership.mem (↑(reesAlgebra I)) ((Polynomial.monomial 1) r)
    -/
    exact reesAlgebra.monomial_mem.mpr (by rwa [pow_one])
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      ⊢ LE.le (reesAlgebra I) (Algebra.adjoin R ↑(Submodule.map (Polynomial.monomial …
    -/
  · intro p hp
    /-
      case a
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      p : Polynomial R
      hp : Membership.mem (reesAlgebra I) p
      ⊢ Membership.mem (Algebra.adjoin R ↑(Submodule.map (Polynomial.monomial 1) I)) p
    -/
    rw [p.as_sum_support]
    /-
      case a
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      p : Polynomial R
      hp : Membership.mem (reesAlgebra I) p
      ⊢ Membership.mem (Algebra.adjoin R ↑(Submodule.map (Polynomial.monomial 1) I)) …
    -/
    apply Subalgebra.sum_mem _ _
    /-
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      p : Polynomial R
      hp : Membership.mem (reesAlgebra I) p
      ⊢ ∀ (x : Nat), Membership.mem p.support x → Membership.mem (Algebra.adjoin R ↑ …
    -/
    rintro i -
    /-
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      p : Polynomial R
      hp : Membership.mem (reesAlgebra I) p
      i : Nat
      ⊢ Membership.mem (Algebra.adjoin R ↑(Submodule.map (Polynomial.monomial 1) I)) …
    -/
    exact monomial_mem_adjoin_monomial (hp i)
    /-
      🎉 no goals
    -/


theorem reesAlgebra.fg (hI : I.FG) : (reesAlgebra I).FG := by
  classical
    obtain ⟨s, hs⟩ := hI
    rw [← adjoin_monomial_eq_reesAlgebra, ← hs]
    use s.image (monomial 1)
    rw [Finset.coe_image]
    change
      _ =
        Algebra.adjoin R
          (Submodule.map (monomial 1 : R →ₗ[R] R[X]) (Submodule.span R ↑s) : Set R[X])
    rw [Submodule.map_span, Algebra.adjoin_span]


instance [IsNoetherianRing R] : Algebra.FiniteType R (reesAlgebra I) :=
  ⟨(reesAlgebra I).fg_top.mpr (reesAlgebra.fg <| IsNoetherian.noetherian I)⟩


instance [IsNoetherianRing R] : IsNoetherianRing (reesAlgebra I) :=
  Algebra.FiniteType.isNoetherianRing R _

