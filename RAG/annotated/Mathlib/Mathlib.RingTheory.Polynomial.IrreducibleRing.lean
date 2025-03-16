/-- A polynomial over an irreducible ring `R` is irreducible if it is monic and irreducible after
mapping into an integral domain `S` (https://math.stackexchange.com/a/4843432/235999).
A generalization to `Polynomial.Monic.irreducible_of_irreducible_map`. -/
theorem Polynomial.Monic.irreducible_of_irreducible_map_of_isPrime_nilradical
    {R S : Type*} [CommRing R] [(nilradical R).IsPrime] [CommRing S] [IsDomain S]
    (φ : R →+* S) (f : R[X]) (hm : f.Monic) (hi : Irreducible (f.map φ)) : Irreducible f := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : (nilradical R).IsPrime
    inst✝¹ : CommRing S
    inst✝ : IsDomain S
    φ : RingHom R S
    f : Polynomial R
    hm : f.Monic
    hi : Irreducible (Polynomial.map φ f)
    ⊢ Irreducible f
  -/
  let R' := R ⧸ nilradical R
  let ψ : R' →+* S := Ideal.Quotient.lift (nilradical R) φ
    (haveI := RingHom.ker_isPrime φ; nilradical_le_prime (RingHom.ker φ))
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : (nilradical R).IsPrime
    inst✝¹ : CommRing S
    inst✝ : IsDomain S
    φ : RingHom R S
    f : Polynomial R
    hm : f.Monic
    hi : Irreducible (Polynomial.map φ f)
    R' : Type u_1 := HasQuotient.Quotient R (nilradical R)
    ψ : RingHom R' S := Ideal.Quotient.lift (nilradical R) φ ⋯
    ⊢ Irreducible f
  -/
  let ι := algebraMap R R'
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : (nilradical R).IsPrime
    inst✝¹ : CommRing S
    inst✝ : IsDomain S
    φ : RingHom R S
    f : Polynomial R
    hm : f.Monic
    hi : Irreducible (Polynomial.map φ f)
    R' : Type u_1 := HasQuotient.Quotient R (nilradical R)
    ψ : RingHom R' S := Ideal.Quotient.lift (nilradical R) φ ⋯
    ι : RingHom R R' := algebraMap R R'
    ⊢ Irreducible f
  -/
  rw [show φ = ψ.comp ι from rfl, ← map_map] at hi
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : (nilradical R).IsPrime
    inst✝¹ : CommRing S
    inst✝ : IsDomain S
    φ : RingHom R S
    f : Polynomial R
    hm : f.Monic
    R' : Type u_1 := HasQuotient.Quotient R (nilradical R)
    ψ : RingHom R' S := Ideal.Quotient.lift (nilradical R) φ ⋯
    ι : RingHom R R' := algebraMap R R'
    hi : Irreducible (Polynomial.map ψ (Polynomial.map ι f))
    ⊢ Irreducible f
  -/
  replace hi := hm.map ι |>.irreducible_of_irreducible_map _ _ hi
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : (nilradical R).IsPrime
    inst✝¹ : CommRing S
    inst✝ : IsDomain S
    φ : RingHom R S
    f : Polynomial R
    hm : f.Monic
    R' : Type u_1 := HasQuotient.Quotient R (nilradical R)
    ψ : RingHom R' S := Ideal.Quotient.lift (nilradical R) φ ⋯
    ι : RingHom R R' := algebraMap R R'
    hi : Irreducible (Polynomial.map ι f)
    ⊢ Irreducible f
  -/
  refine ⟨fun h ↦ hi.1 <| (mapRingHom ι).isUnit_map h, fun a b h ↦ ?_⟩
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : (nilradical R).IsPrime
    inst✝¹ : CommRing S
    inst✝ : IsDomain S
    φ : RingHom R S
    f : Polynomial R
    hm : f.Monic
    R' : Type u_1 := HasQuotient.Quotient R (nilradical R)
    ψ : RingHom R' S := Ideal.Quotient.lift (nilradical R) φ ⋯
    ι : RingHom R R' := algebraMap R R'
    hi : Irreducible (Polynomial.map ι f)
    a b : Polynomial R
    h : Eq f (HMul.hMul a b)
    ⊢ Or (IsUnit a) (IsUnit b)
  -/
  wlog hb : IsUnit (b.map ι) generalizing a b
  · exact (this b a (mul_comm a b ▸ h)
      (hi.2 _ _ (by rw [h, Polynomial.map_mul]) |>.resolve_right hb)).symm
  have hn (i : ℕ) (hi : i ≠ 0) : IsNilpotent (b.coeff i) := by
    obtain ⟨_, _, h⟩ := Polynomial.isUnit_iff.1 hb
    simpa only [coeff_map, coeff_C, hi, ite_false, ← RingHom.mem_ker,
      show RingHom.ker ι = nilradical R from Ideal.mk_ker] using congr(coeff $(h.symm) i)
  refine .inr <| isUnit_of_coeff_isUnit_isNilpotent (isUnit_of_mul_isUnit_right
    (x := a.coeff f.natDegree) <| (IsUnit.neg_iff _).1 ?_) hn
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : (nilradical R).IsPrime
    inst✝¹ : CommRing S
    inst✝ : IsDomain S
    φ : RingHom R S
    f : Polynomial R
    hm : f.Monic
    R' : Type u_1 := HasQuotient.Quotient R (nilradical R)
    ψ : RingHom R' S := Ideal.Quotient.lift (nilradical R) φ ⋯
    ι : RingHom R R' := algebraMap R R'
    hi : Irreducible (Polynomial.map ι f)
    a b : Polynomial R
    h : Eq f (HMul.hMul a b)
    hb : IsUnit (Polynomial.map ι b)
    hn : ∀ (i : Nat), Ne i 0 → IsNilpotent (b.coeff i)
    ⊢ IsUnit (Neg.neg (HMul.hMul (a.coeff f.natDegree) (b.coeff 0)))
  -/
  have hc : f.leadingCoeff = _ := congr(coeff $h f.natDegree)
  rw [hm, coeff_mul, Finset.Nat.sum_antidiagonal_eq_sum_range_succ fun i j ↦ a.coeff i * b.coeff j,
    Finset.sum_range_succ, ← sub_eq_iff_eq_add, Nat.sub_self] at hc
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : (nilradical R).IsPrime
    inst✝¹ : CommRing S
    inst✝ : IsDomain S
    φ : RingHom R S
    f : Polynomial R
    hm : f.Monic
    R' : Type u_1 := HasQuotient.Quotient R (nilradical R)
    ψ : RingHom R' S := Ideal.Quotient.lift (nilradical R) φ ⋯
    ι : RingHom R R' := algebraMap R R'
    hi : Irreducible (Polynomial.map ι f)
    a b : Polynomial R
    h : Eq f (HMul.hMul a b)
    hb : IsUnit (Polynomial.map ι b)
    hn : ∀ (i : Nat), Ne i 0 → IsNilpotent (b.coeff i)
    hc : Eq (HSub.hSub 1 (HMul.hMul (a.coeff f.natDegree) (b.coeff 0))) ((Finset.r …
    ⊢ IsUnit (Neg.neg (HMul.hMul (a.coeff f.natDegree) (b.coeff 0)))
  -/
  rw [← add_sub_cancel_left 1 (-(_ * _)), ← sub_eq_add_neg, hc]
  exact IsNilpotent.isUnit_sub_one <| show _ ∈ nilradical R from sum_mem fun i hi ↦
    Ideal.mul_mem_left _ _ <| hn _ <| Nat.sub_ne_zero_of_lt (List.mem_range.1 hi)

