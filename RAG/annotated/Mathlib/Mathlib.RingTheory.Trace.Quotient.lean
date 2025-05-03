local notation "p" => maximalIdeal R

local notation "pS" => Ideal.map (algebraMap R S) p


lemma Algebra.trace_quotient_mk [IsLocalRing R] (x : S) :
    Algebra.trace (R ⧸ p) (S ⧸ pS) (Ideal.Quotient.mk pS x) =
      Ideal.Quotient.mk p (Algebra.trace R S x) := by
  classical
  let ι := Module.Free.ChooseBasisIndex R S
  let b : Basis ι R S := Module.Free.chooseBasis R S
  rw [trace_eq_matrix_trace b, trace_eq_matrix_trace (basisQuotient b), AddMonoidHom.map_trace]
  congr 1
  ext i j
  simp only [leftMulMatrix_apply, coe_lmul_eq_mul, LinearMap.toMatrix_apply,
    basisQuotient_apply, LinearMap.mul_apply', RingHom.toAddMonoidHom_eq_coe,
    AddMonoidHom.mapMatrix_apply, AddMonoidHom.coe_coe, Matrix.map_apply, ← map_mul,
    basisQuotient_repr]


/-- The isomorphism `R ⧸ p ≃+* Rₚ ⧸ maximalIdeal Rₚ`, where `Rₚ` satisfies
`IsLocalization.AtPrime Rₚ p`. In particular, localization preserves the residue field. -/
noncomputable
def equivQuotMaximalIdealOfIsLocalization : R ⧸ p ≃+* Rₚ ⧸ maximalIdeal Rₚ := by
  refine (Ideal.quotEquivOfEq ?_).trans
    (RingHom.quotientKerEquivOfSurjective (f := algebraMap R (Rₚ ⧸ maximalIdeal Rₚ)) ?_)
  · rw [IsScalarTower.algebraMap_eq R Rₚ, ← RingHom.comap_ker,
      Ideal.Quotient.algebraMap_eq, Ideal.mk_ker, IsLocalization.AtPrime.comap_maximalIdeal Rₚ p]
    /-
      case refine_2
      R : Type u_1
      S : Type u_2
      inst✝¹⁴ : CommRing R
      inst✝¹³ : CommRing S
      inst✝¹² : Algebra R S
      p : Ideal R
      inst✝¹¹ : p.IsMaximal
      Rₚ : Type u_3
      Sₚ : Type u_4
      inst✝¹⁰ : CommRing Rₚ
      inst✝⁹ : CommRing Sₚ
      inst✝⁸ : Algebra R Rₚ
      inst✝⁷ : IsLocalization.AtPrime Rₚ p
      inst✝⁶ : IsLocalRing Rₚ
      inst✝⁵ : Algebra S Sₚ
      inst✝⁴ : Algebra R Sₚ
      inst✝³ : Algebra Rₚ Sₚ
      inst✝² : IsLocalization (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ
      inst✝¹ : IsScalarTower R S Sₚ
      inst✝ : IsScalarTower R Rₚ Sₚ
      ⊢ Function.Surjective ⇑(algebraMap R (HasQuotient.Quotient Rₚ (IsLocalRing.max …
    -/
  · intro x
    /-
      case refine_2
      R : Type u_1
      S : Type u_2
      inst✝¹⁴ : CommRing R
      inst✝¹³ : CommRing S
      inst✝¹² : Algebra R S
      p : Ideal R
      inst✝¹¹ : p.IsMaximal
      Rₚ : Type u_3
      Sₚ : Type u_4
      inst✝¹⁰ : CommRing Rₚ
      inst✝⁹ : CommRing Sₚ
      inst✝⁸ : Algebra R Rₚ
      inst✝⁷ : IsLocalization.AtPrime Rₚ p
      inst✝⁶ : IsLocalRing Rₚ
      inst✝⁵ : Algebra S Sₚ
      inst✝⁴ : Algebra R Sₚ
      inst✝³ : Algebra Rₚ Sₚ
      inst✝² : IsLocalization (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ
      inst✝¹ : IsScalarTower R S Sₚ
      inst✝ : IsScalarTower R Rₚ Sₚ
      x : HasQuotient.Quotient Rₚ (IsLocalRing.maximalIdeal Rₚ)
      ⊢ Exists fun a => Eq ((algebraMap R (HasQuotient.Quotient Rₚ (IsLocalRing.maxi …
    -/
    obtain ⟨x, rfl⟩ := Ideal.Quotient.mk_surjective x
    /-
      case refine_2.intro
      R : Type u_1
      S : Type u_2
      inst✝¹⁴ : CommRing R
      inst✝¹³ : CommRing S
      inst✝¹² : Algebra R S
      p : Ideal R
      inst✝¹¹ : p.IsMaximal
      Rₚ : Type u_3
      Sₚ : Type u_4
      inst✝¹⁰ : CommRing Rₚ
      inst✝⁹ : CommRing Sₚ
      inst✝⁸ : Algebra R Rₚ
      inst✝⁷ : IsLocalization.AtPrime Rₚ p
      inst✝⁶ : IsLocalRing Rₚ
      inst✝⁵ : Algebra S Sₚ
      inst✝⁴ : Algebra R Sₚ
      inst✝³ : Algebra Rₚ Sₚ
      inst✝² : IsLocalization (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ
      inst✝¹ : IsScalarTower R S Sₚ
      inst✝ : IsScalarTower R Rₚ Sₚ
      x : Rₚ
      ⊢ Exists fun a => Eq ((algebraMap R (HasQuotient.Quotient Rₚ (IsLocalRing.maxi …
    -/
    obtain ⟨x, s, rfl⟩ := IsLocalization.mk'_surjective p.primeCompl x
    /-
      case refine_2.intro.intro.intro
      R : Type u_1
      S : Type u_2
      inst✝¹⁴ : CommRing R
      inst✝¹³ : CommRing S
      inst✝¹² : Algebra R S
      p : Ideal R
      inst✝¹¹ : p.IsMaximal
      Rₚ : Type u_3
      Sₚ : Type u_4
      inst✝¹⁰ : CommRing Rₚ
      inst✝⁹ : CommRing Sₚ
      inst✝⁸ : Algebra R Rₚ
      inst✝⁷ : IsLocalization.AtPrime Rₚ p
      inst✝⁶ : IsLocalRing Rₚ
      inst✝⁵ : Algebra S Sₚ
      inst✝⁴ : Algebra R Sₚ
      inst✝³ : Algebra Rₚ Sₚ
      inst✝² : IsLocalization (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ
      inst✝¹ : IsScalarTower R S Sₚ
      inst✝ : IsScalarTower R Rₚ Sₚ
      x : R
      s : Subtype fun x => Membership.mem p.primeCompl x
      ⊢ Exists fun a => Eq ((algebraMap R (HasQuotient.Quotient Rₚ (IsLocalRing.maxi …
    -/
    obtain ⟨s', hs⟩ := Ideal.Quotient.mk_surjective (I := p) (Ideal.Quotient.mk p s)⁻¹
    simp only [IsScalarTower.algebraMap_eq R Rₚ (Rₚ ⧸ _),
      Ideal.Quotient.algebraMap_eq, RingHom.comp_apply]
    /-
      case refine_2.intro.intro.intro.intro
      R : Type u_1
      S : Type u_2
      inst✝¹⁴ : CommRing R
      inst✝¹³ : CommRing S
      inst✝¹² : Algebra R S
      p : Ideal R
      inst✝¹¹ : p.IsMaximal
      Rₚ : Type u_3
      Sₚ : Type u_4
      inst✝¹⁰ : CommRing Rₚ
      inst✝⁹ : CommRing Sₚ
      inst✝⁸ : Algebra R Rₚ
      inst✝⁷ : IsLocalization.AtPrime Rₚ p
      inst✝⁶ : IsLocalRing Rₚ
      inst✝⁵ : Algebra S Sₚ
      inst✝⁴ : Algebra R Sₚ
      inst✝³ : Algebra Rₚ Sₚ
      inst✝² : IsLocalization (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ
      inst✝¹ : IsScalarTower R S Sₚ
      inst✝ : IsScalarTower R Rₚ Sₚ
      x : R
      s : Subtype fun x => Membership.mem p.primeCompl x
      s' : R
      hs : Eq ((Ideal.Quotient.mk p) s') (Inv.inv ((Ideal.Quotient.mk p) ↑s))
      ⊢ Exists fun a => Eq ((Ideal.Quotient.mk (IsLocalRing.maximalIdeal Rₚ)) ((alge …
    -/
    use x * s'
    /-
      case h
      R : Type u_1
      S : Type u_2
      inst✝¹⁴ : CommRing R
      inst✝¹³ : CommRing S
      inst✝¹² : Algebra R S
      p : Ideal R
      inst✝¹¹ : p.IsMaximal
      Rₚ : Type u_3
      Sₚ : Type u_4
      inst✝¹⁰ : CommRing Rₚ
      inst✝⁹ : CommRing Sₚ
      inst✝⁸ : Algebra R Rₚ
      inst✝⁷ : IsLocalization.AtPrime Rₚ p
      inst✝⁶ : IsLocalRing Rₚ
      inst✝⁵ : Algebra S Sₚ
      inst✝⁴ : Algebra R Sₚ
      inst✝³ : Algebra Rₚ Sₚ
      inst✝² : IsLocalization (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ
      inst✝¹ : IsScalarTower R S Sₚ
      inst✝ : IsScalarTower R Rₚ Sₚ
      x : R
      s : Subtype fun x => Membership.mem p.primeCompl x
      s' : R
      hs : Eq ((Ideal.Quotient.mk p) s') (Inv.inv ((Ideal.Quotient.mk p) ↑s))
      ⊢ Eq ((Ideal.Quotient.mk (IsLocalRing.maximalIdeal Rₚ)) ((algebraMap R Rₚ) (HM …
    -/
    rw [← sub_eq_zero, ← map_sub, Ideal.Quotient.eq_zero_iff_mem]
    have : algebraMap R Rₚ s ∉ maximalIdeal Rₚ := by
      rw [← Ideal.mem_comap, IsLocalization.AtPrime.comap_maximalIdeal Rₚ p]
      exact s.prop
    /-
      case h
      R : Type u_1
      S : Type u_2
      inst✝¹⁴ : CommRing R
      inst✝¹³ : CommRing S
      inst✝¹² : Algebra R S
      p : Ideal R
      inst✝¹¹ : p.IsMaximal
      Rₚ : Type u_3
      Sₚ : Type u_4
      inst✝¹⁰ : CommRing Rₚ
      inst✝⁹ : CommRing Sₚ
      inst✝⁸ : Algebra R Rₚ
      inst✝⁷ : IsLocalization.AtPrime Rₚ p
      inst✝⁶ : IsLocalRing Rₚ
      inst✝⁵ : Algebra S Sₚ
      inst✝⁴ : Algebra R Sₚ
      inst✝³ : Algebra Rₚ Sₚ
      inst✝² : IsLocalization (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ
      inst✝¹ : IsScalarTower R S Sₚ
      inst✝ : IsScalarTower R Rₚ Sₚ
      x : R
      s : Subtype fun x => Membership.mem p.primeCompl x
      s' : R
      hs : Eq ((Ideal.Quotient.mk p) s') (Inv.inv ((Ideal.Quotient.mk p) ↑s))
      this : Not (Membership.mem (IsLocalRing.maximalIdeal Rₚ) ((algebraMap R Rₚ) ↑s))
      ⊢ Membership.mem (IsLocalRing.maximalIdeal Rₚ) (HSub.hSub ((algebraMap R Rₚ) ( …
    -/
    refine ((inferInstanceAs <| (maximalIdeal Rₚ).IsPrime).mem_or_mem ?_).resolve_left this
    rw [mul_sub, IsLocalization.mul_mk'_eq_mk'_of_mul, IsLocalization.mk'_mul_cancel_left,
      ← map_mul, ← map_sub, ← Ideal.mem_comap, IsLocalization.AtPrime.comap_maximalIdeal Rₚ p,
      mul_left_comm, ← Ideal.Quotient.eq_zero_iff_mem, map_sub, map_mul, map_mul, hs,
      mul_inv_cancel₀, mul_one, sub_self]
    /-
      case h
      R : Type u_1
      S : Type u_2
      inst✝¹⁴ : CommRing R
      inst✝¹³ : CommRing S
      inst✝¹² : Algebra R S
      p : Ideal R
      inst✝¹¹ : p.IsMaximal
      Rₚ : Type u_3
      Sₚ : Type u_4
      inst✝¹⁰ : CommRing Rₚ
      inst✝⁹ : CommRing Sₚ
      inst✝⁸ : Algebra R Rₚ
      inst✝⁷ : IsLocalization.AtPrime Rₚ p
      inst✝⁶ : IsLocalRing Rₚ
      inst✝⁵ : Algebra S Sₚ
      inst✝⁴ : Algebra R Sₚ
      inst✝³ : Algebra Rₚ Sₚ
      inst✝² : IsLocalization (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ
      inst✝¹ : IsScalarTower R S Sₚ
      inst✝ : IsScalarTower R Rₚ Sₚ
      x : R
      s : Subtype fun x => Membership.mem p.primeCompl x
      s' : R
      hs : Eq ((Ideal.Quotient.mk p) s') (Inv.inv ((Ideal.Quotient.mk p) ↑s))
      this : Not (Membership.mem (IsLocalRing.maximalIdeal Rₚ) ((algebraMap R Rₚ) ↑s))
      ⊢ Ne ((Ideal.Quotient.mk p) ↑s) 0
    -/
    rw [Ne, Ideal.Quotient.eq_zero_iff_mem]
    /-
      case h
      R : Type u_1
      S : Type u_2
      inst✝¹⁴ : CommRing R
      inst✝¹³ : CommRing S
      inst✝¹² : Algebra R S
      p : Ideal R
      inst✝¹¹ : p.IsMaximal
      Rₚ : Type u_3
      Sₚ : Type u_4
      inst✝¹⁰ : CommRing Rₚ
      inst✝⁹ : CommRing Sₚ
      inst✝⁸ : Algebra R Rₚ
      inst✝⁷ : IsLocalization.AtPrime Rₚ p
      inst✝⁶ : IsLocalRing Rₚ
      inst✝⁵ : Algebra S Sₚ
      inst✝⁴ : Algebra R Sₚ
      inst✝³ : Algebra Rₚ Sₚ
      inst✝² : IsLocalization (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ
      inst✝¹ : IsScalarTower R S Sₚ
      inst✝ : IsScalarTower R Rₚ Sₚ
      x : R
      s : Subtype fun x => Membership.mem p.primeCompl x
      s' : R
      hs : Eq ((Ideal.Quotient.mk p) s') (Inv.inv ((Ideal.Quotient.mk p) ↑s))
      this : Not (Membership.mem (IsLocalRing.maximalIdeal Rₚ) ((algebraMap R Rₚ) ↑s))
      ⊢ Not (Membership.mem p ↑s)
    -/
    exact s.prop
    /-
      🎉 no goals
    -/


lemma IsLocalization.AtPrime.map_eq_maximalIdeal :
    p.map (algebraMap R Rₚ) = maximalIdeal Rₚ := by
  convert congr_arg (Ideal.map (algebraMap R Rₚ))
    (IsLocalization.AtPrime.comap_maximalIdeal Rₚ p).symm
  /-
    case h.e'_3
    R : Type u_1
    inst✝⁵ : CommRing R
    p : Ideal R
    inst✝⁴ : p.IsMaximal
    Rₚ : Type u_3
    inst✝³ : CommRing Rₚ
    inst✝² : Algebra R Rₚ
    inst✝¹ : IsLocalization.AtPrime Rₚ p
    inst✝ : IsLocalRing Rₚ
    ⊢ Eq (IsLocalRing.maximalIdeal Rₚ) (Ideal.map (algebraMap R Rₚ) (Ideal.comap ( …
  -/
  rw [map_comap p.primeCompl]
  /-
    🎉 no goals
  -/


local notation "pS" => Ideal.map (algebraMap R S) p

local notation "pSₚ" => Ideal.map (algebraMap Rₚ Sₚ) (maximalIdeal Rₚ)


lemma comap_map_eq_map_of_isLocalization_algebraMapSubmonoid :
    (Ideal.map (algebraMap R Sₚ) p).comap (algebraMap S Sₚ) = pS := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    p : Ideal R
    inst✝⁵ : p.IsMaximal
    Sₚ : Type u_4
    inst✝⁴ : CommRing Sₚ
    inst✝³ : Algebra S Sₚ
    inst✝² : Algebra R Sₚ
    inst✝¹ : IsLocalization (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ
    inst✝ : IsScalarTower R S Sₚ
    ⊢ Eq (Ideal.comap (algebraMap S Sₚ) (Ideal.map (algebraMap R Sₚ) p)) (Ideal.ma …
  -/
  rw [IsScalarTower.algebraMap_eq R S Sₚ, ← Ideal.map_map, eq_comm]
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    p : Ideal R
    inst✝⁵ : p.IsMaximal
    Sₚ : Type u_4
    inst✝⁴ : CommRing Sₚ
    inst✝³ : Algebra S Sₚ
    inst✝² : Algebra R Sₚ
    inst✝¹ : IsLocalization (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ
    inst✝ : IsScalarTower R S Sₚ
    ⊢ Eq (Ideal.map (algebraMap R S) p) (Ideal.comap (algebraMap S Sₚ) (Ideal.map  …
  -/
  apply Ideal.le_comap_map.antisymm
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    p : Ideal R
    inst✝⁵ : p.IsMaximal
    Sₚ : Type u_4
    inst✝⁴ : CommRing Sₚ
    inst✝³ : Algebra S Sₚ
    inst✝² : Algebra R Sₚ
    inst✝¹ : IsLocalization (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ
    inst✝ : IsScalarTower R S Sₚ
    ⊢ LE.le (Ideal.comap (algebraMap S Sₚ) (Ideal.map (algebraMap S Sₚ) (Ideal.map …
  -/
  intro x hx
  obtain ⟨α, hα, hαx⟩ : ∃ α ∉ p, α • x ∈ pS := by
    have ⟨⟨y, s⟩, hy⟩ := (IsLocalization.mem_map_algebraMap_iff
      (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ).mp hx
    rw [← map_mul,
      IsLocalization.eq_iff_exists (Algebra.algebraMapSubmonoid S p.primeCompl)] at hy
    obtain ⟨c, hc⟩ := hy
    obtain ⟨α, hα, e⟩ := (c * s).prop
    refine ⟨α, hα, ?_⟩
    rw [Algebra.smul_def, e, Submonoid.coe_mul, mul_assoc, mul_comm _ x, hc]
    exact Ideal.mul_mem_left _ _ y.prop
  obtain ⟨β, γ, hγ, hβ⟩ : ∃ β γ, γ ∈ p ∧ β * α = 1 + γ := by
    obtain ⟨β, hβ⟩ := Ideal.Quotient.mk_surjective (I := p) (Ideal.Quotient.mk p α)⁻¹
    refine ⟨β, β * α - 1, ?_, ?_⟩
    · rw [← Ideal.Quotient.eq_zero_iff_mem, map_sub, map_one,
        map_mul, hβ, inv_mul_cancel₀, sub_self]
      rwa [Ne, Ideal.Quotient.eq_zero_iff_mem]
    · rw [add_sub_cancel]
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    p : Ideal R
    inst✝⁵ : p.IsMaximal
    Sₚ : Type u_4
    inst✝⁴ : CommRing Sₚ
    inst✝³ : Algebra S Sₚ
    inst✝² : Algebra R Sₚ
    inst✝¹ : IsLocalization (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ
    inst✝ : IsScalarTower R S Sₚ
    x : S
    hx : Membership.mem (Ideal.comap (algebraMap S Sₚ) (Ideal.map (algebraMap S Sₚ …
    α : R
    hα : Not (Membership.mem p α)
    hαx : Membership.mem (Ideal.map (algebraMap R S) p) (HSMul.hSMul α x)
    β γ : R
    hγ : Membership.mem p γ
    hβ : Eq (HMul.hMul β α) (HAdd.hAdd 1 γ)
    ⊢ Membership.mem (Ideal.map (algebraMap R S) p) x
  -/
  have := Ideal.mul_mem_left _ (algebraMap _ _ β) hαx
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    p : Ideal R
    inst✝⁵ : p.IsMaximal
    Sₚ : Type u_4
    inst✝⁴ : CommRing Sₚ
    inst✝³ : Algebra S Sₚ
    inst✝² : Algebra R Sₚ
    inst✝¹ : IsLocalization (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ
    inst✝ : IsScalarTower R S Sₚ
    x : S
    hx : Membership.mem (Ideal.comap (algebraMap S Sₚ) (Ideal.map (algebraMap S Sₚ …
    α : R
    hα : Not (Membership.mem p α)
    hαx : Membership.mem (Ideal.map (algebraMap R S) p) (HSMul.hSMul α x)
    β γ : R
    hγ : Membership.mem p γ
    hβ : Eq (HMul.hMul β α) (HAdd.hAdd 1 γ)
    this : Membership.mem (Ideal.map (algebraMap R S) p) (HMul.hMul ((algebraMap R …
    ⊢ Membership.mem (Ideal.map (algebraMap R S) p) x
  -/
  rw [← Algebra.smul_def, smul_smul, hβ, add_smul, one_smul] at this
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    p : Ideal R
    inst✝⁵ : p.IsMaximal
    Sₚ : Type u_4
    inst✝⁴ : CommRing Sₚ
    inst✝³ : Algebra S Sₚ
    inst✝² : Algebra R Sₚ
    inst✝¹ : IsLocalization (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ
    inst✝ : IsScalarTower R S Sₚ
    x : S
    hx : Membership.mem (Ideal.comap (algebraMap S Sₚ) (Ideal.map (algebraMap S Sₚ …
    α : R
    hα : Not (Membership.mem p α)
    hαx : Membership.mem (Ideal.map (algebraMap R S) p) (HSMul.hSMul α x)
    β γ : R
    hγ : Membership.mem p γ
    hβ : Eq (HMul.hMul β α) (HAdd.hAdd 1 γ)
    this : Membership.mem (Ideal.map (algebraMap R S) p) (HAdd.hAdd x (HSMul.hSMul …
    ⊢ Membership.mem (Ideal.map (algebraMap R S) p) x
  -/
  refine (Submodule.add_mem_iff_left _ ?_).mp this
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    p : Ideal R
    inst✝⁵ : p.IsMaximal
    Sₚ : Type u_4
    inst✝⁴ : CommRing Sₚ
    inst✝³ : Algebra S Sₚ
    inst✝² : Algebra R Sₚ
    inst✝¹ : IsLocalization (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ
    inst✝ : IsScalarTower R S Sₚ
    x : S
    hx : Membership.mem (Ideal.comap (algebraMap S Sₚ) (Ideal.map (algebraMap S Sₚ …
    α : R
    hα : Not (Membership.mem p α)
    hαx : Membership.mem (Ideal.map (algebraMap R S) p) (HSMul.hSMul α x)
    β γ : R
    hγ : Membership.mem p γ
    hβ : Eq (HMul.hMul β α) (HAdd.hAdd 1 γ)
    this : Membership.mem (Ideal.map (algebraMap R S) p) (HAdd.hAdd x (HSMul.hSMul …
    ⊢ Membership.mem (Ideal.map (algebraMap R S) p) (HSMul.hSMul γ x)
  -/
  rw [Algebra.smul_def]
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    p : Ideal R
    inst✝⁵ : p.IsMaximal
    Sₚ : Type u_4
    inst✝⁴ : CommRing Sₚ
    inst✝³ : Algebra S Sₚ
    inst✝² : Algebra R Sₚ
    inst✝¹ : IsLocalization (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ
    inst✝ : IsScalarTower R S Sₚ
    x : S
    hx : Membership.mem (Ideal.comap (algebraMap S Sₚ) (Ideal.map (algebraMap S Sₚ …
    α : R
    hα : Not (Membership.mem p α)
    hαx : Membership.mem (Ideal.map (algebraMap R S) p) (HSMul.hSMul α x)
    β γ : R
    hγ : Membership.mem p γ
    hβ : Eq (HMul.hMul β α) (HAdd.hAdd 1 γ)
    this : Membership.mem (Ideal.map (algebraMap R S) p) (HAdd.hAdd x (HSMul.hSMul …
    ⊢ Membership.mem (Ideal.map (algebraMap R S) p) (HMul.hMul ((algebraMap R S) γ …
  -/
  apply Ideal.mul_mem_right
  /-
    case intro.intro.intro.intro.intro.h
    R : Type u_1
    S : Type u_2
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    p : Ideal R
    inst✝⁵ : p.IsMaximal
    Sₚ : Type u_4
    inst✝⁴ : CommRing Sₚ
    inst✝³ : Algebra S Sₚ
    inst✝² : Algebra R Sₚ
    inst✝¹ : IsLocalization (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ
    inst✝ : IsScalarTower R S Sₚ
    x : S
    hx : Membership.mem (Ideal.comap (algebraMap S Sₚ) (Ideal.map (algebraMap S Sₚ …
    α : R
    hα : Not (Membership.mem p α)
    hαx : Membership.mem (Ideal.map (algebraMap R S) p) (HSMul.hSMul α x)
    β γ : R
    hγ : Membership.mem p γ
    hβ : Eq (HMul.hMul β α) (HAdd.hAdd 1 γ)
    this : Membership.mem (Ideal.map (algebraMap R S) p) (HAdd.hAdd x (HSMul.hSMul …
    ⊢ Membership.mem (Ideal.map (algebraMap R S) p) ((algebraMap R S) γ)
  -/
  exact Ideal.mem_map_of_mem _ hγ
  /-
    🎉 no goals
  -/


/-- The isomorphism `S ⧸ pS ≃+* Sₚ ⧸ pSₚ`. -/
noncomputable
def quotMapEquivQuotMapMaximalIdealOfIsLocalization : S ⧸ pS ≃+* Sₚ ⧸ pSₚ := by
  haveI h : pSₚ = Ideal.map (algebraMap S Sₚ) pS := by
    rw [← IsLocalization.AtPrime.map_eq_maximalIdeal p Rₚ, Ideal.map_map,
      ← IsScalarTower.algebraMap_eq, Ideal.map_map, ← IsScalarTower.algebraMap_eq]
  refine (Ideal.quotEquivOfEq ?_).trans
    (RingHom.quotientKerEquivOfSurjective (f := algebraMap S (Sₚ ⧸ pSₚ)) ?_)
  · rw [IsScalarTower.algebraMap_eq S Sₚ, Ideal.Quotient.algebraMap_eq, ← RingHom.comap_ker,
      Ideal.mk_ker, h, Ideal.map_map, ← IsScalarTower.algebraMap_eq,
      comap_map_eq_map_of_isLocalization_algebraMapSubmonoid]
    /-
      case refine_2
      R : Type u_1
      S : Type u_2
      inst✝¹⁴ : CommRing R
      inst✝¹³ : CommRing S
      inst✝¹² : Algebra R S
      p : Ideal R
      inst✝¹¹ : p.IsMaximal
      Rₚ : Type u_3
      Sₚ : Type u_4
      inst✝¹⁰ : CommRing Rₚ
      inst✝⁹ : CommRing Sₚ
      inst✝⁸ : Algebra R Rₚ
      inst✝⁷ : IsLocalization.AtPrime Rₚ p
      inst✝⁶ : IsLocalRing Rₚ
      inst✝⁵ : Algebra S Sₚ
      inst✝⁴ : Algebra R Sₚ
      inst✝³ : Algebra Rₚ Sₚ
      inst✝² : IsLocalization (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ
      inst✝¹ : IsScalarTower R S Sₚ
      inst✝ : IsScalarTower R Rₚ Sₚ
      h : Eq (Ideal.map (algebraMap Rₚ Sₚ) (IsLocalRing.maximalIdeal Rₚ)) (Ideal.map …
      ⊢ Function.Surjective ⇑(algebraMap S (HasQuotient.Quotient Sₚ (Ideal.map (alge …
    -/
  · intro x
    /-
      case refine_2
      R : Type u_1
      S : Type u_2
      inst✝¹⁴ : CommRing R
      inst✝¹³ : CommRing S
      inst✝¹² : Algebra R S
      p : Ideal R
      inst✝¹¹ : p.IsMaximal
      Rₚ : Type u_3
      Sₚ : Type u_4
      inst✝¹⁰ : CommRing Rₚ
      inst✝⁹ : CommRing Sₚ
      inst✝⁸ : Algebra R Rₚ
      inst✝⁷ : IsLocalization.AtPrime Rₚ p
      inst✝⁶ : IsLocalRing Rₚ
      inst✝⁵ : Algebra S Sₚ
      inst✝⁴ : Algebra R Sₚ
      inst✝³ : Algebra Rₚ Sₚ
      inst✝² : IsLocalization (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ
      inst✝¹ : IsScalarTower R S Sₚ
      inst✝ : IsScalarTower R Rₚ Sₚ
      h : Eq (Ideal.map (algebraMap Rₚ Sₚ) (IsLocalRing.maximalIdeal Rₚ)) (Ideal.map …
      x : HasQuotient.Quotient Sₚ (Ideal.map (algebraMap Rₚ Sₚ) (IsLocalRing.maximal …
      ⊢ Exists fun a => Eq ((algebraMap S (HasQuotient.Quotient Sₚ (Ideal.map (algeb …
    -/
    obtain ⟨x, rfl⟩ := Ideal.Quotient.mk_surjective x
    obtain ⟨x, s, rfl⟩ := IsLocalization.mk'_surjective
      (Algebra.algebraMapSubmonoid S p.primeCompl) x
    /-
      case refine_2.intro.intro.intro
      R : Type u_1
      S : Type u_2
      inst✝¹⁴ : CommRing R
      inst✝¹³ : CommRing S
      inst✝¹² : Algebra R S
      p : Ideal R
      inst✝¹¹ : p.IsMaximal
      Rₚ : Type u_3
      Sₚ : Type u_4
      inst✝¹⁰ : CommRing Rₚ
      inst✝⁹ : CommRing Sₚ
      inst✝⁸ : Algebra R Rₚ
      inst✝⁷ : IsLocalization.AtPrime Rₚ p
      inst✝⁶ : IsLocalRing Rₚ
      inst✝⁵ : Algebra S Sₚ
      inst✝⁴ : Algebra R Sₚ
      inst✝³ : Algebra Rₚ Sₚ
      inst✝² : IsLocalization (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ
      inst✝¹ : IsScalarTower R S Sₚ
      inst✝ : IsScalarTower R Rₚ Sₚ
      h : Eq (Ideal.map (algebraMap Rₚ Sₚ) (IsLocalRing.maximalIdeal Rₚ)) (Ideal.map …
      x : S
      s : Subtype fun x => Membership.mem (Algebra.algebraMapSubmonoid S p.primeComp …
      ⊢ Exists fun a => Eq ((algebraMap S (HasQuotient.Quotient Sₚ (Ideal.map (algeb …
    -/
    obtain ⟨α, hα : α ∉ p, e⟩ := s.prop
    obtain ⟨β, γ, hγ, hβ⟩ : ∃ β γ, γ ∈ p ∧ α * β = 1 + γ := by
      obtain ⟨β, hβ⟩ := Ideal.Quotient.mk_surjective (I := p) (Ideal.Quotient.mk p α)⁻¹
      refine ⟨β, α * β - 1, ?_, ?_⟩
      · rw [← Ideal.Quotient.eq_zero_iff_mem, map_sub, map_one,
          map_mul, hβ, mul_inv_cancel₀, sub_self]
        rwa [Ne, Ideal.Quotient.eq_zero_iff_mem]
      · rw [add_sub_cancel]
    /-
      case refine_2.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u_1
      S : Type u_2
      inst✝¹⁴ : CommRing R
      inst✝¹³ : CommRing S
      inst✝¹² : Algebra R S
      p : Ideal R
      inst✝¹¹ : p.IsMaximal
      Rₚ : Type u_3
      Sₚ : Type u_4
      inst✝¹⁰ : CommRing Rₚ
      inst✝⁹ : CommRing Sₚ
      inst✝⁸ : Algebra R Rₚ
      inst✝⁷ : IsLocalization.AtPrime Rₚ p
      inst✝⁶ : IsLocalRing Rₚ
      inst✝⁵ : Algebra S Sₚ
      inst✝⁴ : Algebra R Sₚ
      inst✝³ : Algebra Rₚ Sₚ
      inst✝² : IsLocalization (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ
      inst✝¹ : IsScalarTower R S Sₚ
      inst✝ : IsScalarTower R Rₚ Sₚ
      h : Eq (Ideal.map (algebraMap Rₚ Sₚ) (IsLocalRing.maximalIdeal Rₚ)) (Ideal.map …
      x : S
      s : Subtype fun x => Membership.mem (Algebra.algebraMapSubmonoid S p.primeComp …
      α : R
      hα : Not (Membership.mem p α)
      e : Eq ((algebraMap R S) α) ↑s
      β γ : R
      hγ : Membership.mem p γ
      hβ : Eq (HMul.hMul α β) (HAdd.hAdd 1 γ)
      ⊢ Exists fun a => Eq ((algebraMap S (HasQuotient.Quotient Sₚ (Ideal.map (algeb …
    -/
    use β • x
    rw [IsScalarTower.algebraMap_eq S Sₚ (Sₚ ⧸ pSₚ), Ideal.Quotient.algebraMap_eq,
      RingHom.comp_apply, ← sub_eq_zero, ← map_sub, Ideal.Quotient.eq_zero_iff_mem]
    rw [h, IsLocalization.mem_map_algebraMap_iff
      (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ]
    /-
      case h
      R : Type u_1
      S : Type u_2
      inst✝¹⁴ : CommRing R
      inst✝¹³ : CommRing S
      inst✝¹² : Algebra R S
      p : Ideal R
      inst✝¹¹ : p.IsMaximal
      Rₚ : Type u_3
      Sₚ : Type u_4
      inst✝¹⁰ : CommRing Rₚ
      inst✝⁹ : CommRing Sₚ
      inst✝⁸ : Algebra R Rₚ
      inst✝⁷ : IsLocalization.AtPrime Rₚ p
      inst✝⁶ : IsLocalRing Rₚ
      inst✝⁵ : Algebra S Sₚ
      inst✝⁴ : Algebra R Sₚ
      inst✝³ : Algebra Rₚ Sₚ
      inst✝² : IsLocalization (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ
      inst✝¹ : IsScalarTower R S Sₚ
      inst✝ : IsScalarTower R Rₚ Sₚ
      h : Eq (Ideal.map (algebraMap Rₚ Sₚ) (IsLocalRing.maximalIdeal Rₚ)) (Ideal.map …
      x : S
      s : Subtype fun x => Membership.mem (Algebra.algebraMapSubmonoid S p.primeComp …
      α : R
      hα : Not (Membership.mem p α)
      e : Eq ((algebraMap R S) α) ↑s
      β γ : R
      hγ : Membership.mem p γ
      hβ : Eq (HMul.hMul α β) (HAdd.hAdd 1 γ)
      ⊢ Exists fun x_1 => Eq (HMul.hMul (HSub.hSub ((algebraMap S Sₚ) (HSMul.hSMul β …
    -/
    refine ⟨⟨⟨γ • x, ?_⟩, s⟩, ?_⟩
      /-
        case h.refine_1
        R : Type u_1
        S : Type u_2
        inst✝¹⁴ : CommRing R
        inst✝¹³ : CommRing S
        inst✝¹² : Algebra R S
        p : Ideal R
        inst✝¹¹ : p.IsMaximal
        Rₚ : Type u_3
        Sₚ : Type u_4
        inst✝¹⁰ : CommRing Rₚ
        inst✝⁹ : CommRing Sₚ
        inst✝⁸ : Algebra R Rₚ
        inst✝⁷ : IsLocalization.AtPrime Rₚ p
        inst✝⁶ : IsLocalRing Rₚ
        inst✝⁵ : Algebra S Sₚ
        inst✝⁴ : Algebra R Sₚ
        inst✝³ : Algebra Rₚ Sₚ
        inst✝² : IsLocalization (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ
        inst✝¹ : IsScalarTower R S Sₚ
        inst✝ : IsScalarTower R Rₚ Sₚ
        h : Eq (Ideal.map (algebraMap Rₚ Sₚ) (IsLocalRing.maximalIdeal Rₚ)) (Ideal.map …
        x : S
        s : Subtype fun x => Membership.mem (Algebra.algebraMapSubmonoid S p.primeComp …
        α : R
        hα : Not (Membership.mem p α)
        e : Eq ((algebraMap R S) α) ↑s
        β γ : R
        hγ : Membership.mem p γ
        hβ : Eq (HMul.hMul α β) (HAdd.hAdd 1 γ)
        ⊢ Membership.mem (Ideal.map (algebraMap R S) p) (HSMul.hSMul γ x)
      -/
    · rw [Algebra.smul_def]
      /-
        case h.refine_1
        R : Type u_1
        S : Type u_2
        inst✝¹⁴ : CommRing R
        inst✝¹³ : CommRing S
        inst✝¹² : Algebra R S
        p : Ideal R
        inst✝¹¹ : p.IsMaximal
        Rₚ : Type u_3
        Sₚ : Type u_4
        inst✝¹⁰ : CommRing Rₚ
        inst✝⁹ : CommRing Sₚ
        inst✝⁸ : Algebra R Rₚ
        inst✝⁷ : IsLocalization.AtPrime Rₚ p
        inst✝⁶ : IsLocalRing Rₚ
        inst✝⁵ : Algebra S Sₚ
        inst✝⁴ : Algebra R Sₚ
        inst✝³ : Algebra Rₚ Sₚ
        inst✝² : IsLocalization (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ
        inst✝¹ : IsScalarTower R S Sₚ
        inst✝ : IsScalarTower R Rₚ Sₚ
        h : Eq (Ideal.map (algebraMap Rₚ Sₚ) (IsLocalRing.maximalIdeal Rₚ)) (Ideal.map …
        x : S
        s : Subtype fun x => Membership.mem (Algebra.algebraMapSubmonoid S p.primeComp …
        α : R
        hα : Not (Membership.mem p α)
        e : Eq ((algebraMap R S) α) ↑s
        β γ : R
        hγ : Membership.mem p γ
        hβ : Eq (HMul.hMul α β) (HAdd.hAdd 1 γ)
        ⊢ Membership.mem (Ideal.map (algebraMap R S) p) (HMul.hMul ((algebraMap R S) γ …
      -/
      apply Ideal.mul_mem_right
      /-
        case h.refine_1.h
        R : Type u_1
        S : Type u_2
        inst✝¹⁴ : CommRing R
        inst✝¹³ : CommRing S
        inst✝¹² : Algebra R S
        p : Ideal R
        inst✝¹¹ : p.IsMaximal
        Rₚ : Type u_3
        Sₚ : Type u_4
        inst✝¹⁰ : CommRing Rₚ
        inst✝⁹ : CommRing Sₚ
        inst✝⁸ : Algebra R Rₚ
        inst✝⁷ : IsLocalization.AtPrime Rₚ p
        inst✝⁶ : IsLocalRing Rₚ
        inst✝⁵ : Algebra S Sₚ
        inst✝⁴ : Algebra R Sₚ
        inst✝³ : Algebra Rₚ Sₚ
        inst✝² : IsLocalization (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ
        inst✝¹ : IsScalarTower R S Sₚ
        inst✝ : IsScalarTower R Rₚ Sₚ
        h : Eq (Ideal.map (algebraMap Rₚ Sₚ) (IsLocalRing.maximalIdeal Rₚ)) (Ideal.map …
        x : S
        s : Subtype fun x => Membership.mem (Algebra.algebraMapSubmonoid S p.primeComp …
        α : R
        hα : Not (Membership.mem p α)
        e : Eq ((algebraMap R S) α) ↑s
        β γ : R
        hγ : Membership.mem p γ
        hβ : Eq (HMul.hMul α β) (HAdd.hAdd 1 γ)
        ⊢ Membership.mem (Ideal.map (algebraMap R S) p) ((algebraMap R S) γ)
      -/
      exact Ideal.mem_map_of_mem _ hγ
      /-
        🎉 no goals
      -/
    /-
      case h.refine_2
      R : Type u_1
      S : Type u_2
      inst✝¹⁴ : CommRing R
      inst✝¹³ : CommRing S
      inst✝¹² : Algebra R S
      p : Ideal R
      inst✝¹¹ : p.IsMaximal
      Rₚ : Type u_3
      Sₚ : Type u_4
      inst✝¹⁰ : CommRing Rₚ
      inst✝⁹ : CommRing Sₚ
      inst✝⁸ : Algebra R Rₚ
      inst✝⁷ : IsLocalization.AtPrime Rₚ p
      inst✝⁶ : IsLocalRing Rₚ
      inst✝⁵ : Algebra S Sₚ
      inst✝⁴ : Algebra R Sₚ
      inst✝³ : Algebra Rₚ Sₚ
      inst✝² : IsLocalization (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ
      inst✝¹ : IsScalarTower R S Sₚ
      inst✝ : IsScalarTower R Rₚ Sₚ
      h : Eq (Ideal.map (algebraMap Rₚ Sₚ) (IsLocalRing.maximalIdeal Rₚ)) (Ideal.map …
      x : S
      s : Subtype fun x => Membership.mem (Algebra.algebraMapSubmonoid S p.primeComp …
      α : R
      hα : Not (Membership.mem p α)
      e : Eq ((algebraMap R S) α) ↑s
      β γ : R
      hγ : Membership.mem p γ
      hβ : Eq (HMul.hMul α β) (HAdd.hAdd 1 γ)
      ⊢ Eq (HMul.hMul (HSub.hSub ((algebraMap S Sₚ) (HSMul.hSMul β x)) (IsLocalizati …
    -/
    simp only
    rw [mul_comm, mul_sub, IsLocalization.mul_mk'_eq_mk'_of_mul,
      IsLocalization.mk'_mul_cancel_left, ← map_mul, ← e, ← Algebra.smul_def, smul_smul,
      hβ, ← map_sub, add_smul, one_smul, add_comm x, add_sub_cancel_right]


lemma trace_quotient_eq_trace_localization_quotient (x) :
    Algebra.trace (R ⧸ p) (S ⧸ pS) (Ideal.Quotient.mk pS x) =
      (equivQuotMaximalIdealOfIsLocalization p Rₚ).symm
        (Algebra.trace (Rₚ ⧸ maximalIdeal Rₚ) (Sₚ ⧸ pSₚ) (algebraMap S _ x)) := by
  have : IsScalarTower R (Rₚ ⧸ maximalIdeal Rₚ) (Sₚ ⧸ pSₚ) := by
    apply IsScalarTower.of_algebraMap_eq'
    rw [IsScalarTower.algebraMap_eq R Rₚ (Rₚ ⧸ _), IsScalarTower.algebraMap_eq R Rₚ (Sₚ ⧸ _),
      ← RingHom.comp_assoc, ← IsScalarTower.algebraMap_eq Rₚ]
  rw [Algebra.trace_eq_of_equiv_equiv (equivQuotMaximalIdealOfIsLocalization p Rₚ)
    (quotMapEquivQuotMapMaximalIdealOfIsLocalization S p Rₚ Sₚ)]
    /-
      R : Type u_1
      S : Type u_2
      inst✝¹⁴ : CommRing R
      inst✝¹³ : CommRing S
      inst✝¹² : Algebra R S
      p : Ideal R
      inst✝¹¹ : p.IsMaximal
      Rₚ : Type u_3
      Sₚ : Type u_4
      inst✝¹⁰ : CommRing Rₚ
      inst✝⁹ : CommRing Sₚ
      inst✝⁸ : Algebra R Rₚ
      inst✝⁷ : IsLocalization.AtPrime Rₚ p
      inst✝⁶ : IsLocalRing Rₚ
      inst✝⁵ : Algebra S Sₚ
      inst✝⁴ : Algebra R Sₚ
      inst✝³ : Algebra Rₚ Sₚ
      inst✝² : IsLocalization (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ
      inst✝¹ : IsScalarTower R S Sₚ
      inst✝ : IsScalarTower R Rₚ Sₚ
      x : S
      this : IsScalarTower R (HasQuotient.Quotient Rₚ (IsLocalRing.maximalIdeal Rₚ)) …
      ⊢ Eq ((equivQuotMaximalIdealOfIsLocalization p Rₚ).symm ((Algebra.trace (HasQu …
    -/
  · congr
    /-
      🎉 no goals
    -/
    /-
      case he
      R : Type u_1
      S : Type u_2
      inst✝¹⁴ : CommRing R
      inst✝¹³ : CommRing S
      inst✝¹² : Algebra R S
      p : Ideal R
      inst✝¹¹ : p.IsMaximal
      Rₚ : Type u_3
      Sₚ : Type u_4
      inst✝¹⁰ : CommRing Rₚ
      inst✝⁹ : CommRing Sₚ
      inst✝⁸ : Algebra R Rₚ
      inst✝⁷ : IsLocalization.AtPrime Rₚ p
      inst✝⁶ : IsLocalRing Rₚ
      inst✝⁵ : Algebra S Sₚ
      inst✝⁴ : Algebra R Sₚ
      inst✝³ : Algebra Rₚ Sₚ
      inst✝² : IsLocalization (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ
      inst✝¹ : IsScalarTower R S Sₚ
      inst✝ : IsScalarTower R Rₚ Sₚ
      x : S
      this : IsScalarTower R (HasQuotient.Quotient Rₚ (IsLocalRing.maximalIdeal Rₚ)) …
      ⊢ Eq ((algebraMap (HasQuotient.Quotient Rₚ (IsLocalRing.maximalIdeal Rₚ)) (Has …
    -/
  · ext x
    simp only [equivQuotMaximalIdealOfIsLocalization, RingHom.quotientKerEquivOfSurjective,
      RingEquiv.coe_ringHom_trans, RingHom.coe_comp, RingHom.coe_coe, Function.comp_apply,
      Ideal.quotEquivOfEq_mk, RingHom.quotientKerEquivOfRightInverse.apply, RingHom.kerLift_mk,
      quotMapEquivQuotMapMaximalIdealOfIsLocalization,
      Ideal.Quotient.algebraMap_quotient_map_quotient]
    /-
      case he.h.a
      R : Type u_1
      S : Type u_2
      inst✝¹⁴ : CommRing R
      inst✝¹³ : CommRing S
      inst✝¹² : Algebra R S
      p : Ideal R
      inst✝¹¹ : p.IsMaximal
      Rₚ : Type u_3
      Sₚ : Type u_4
      inst✝¹⁰ : CommRing Rₚ
      inst✝⁹ : CommRing Sₚ
      inst✝⁸ : Algebra R Rₚ
      inst✝⁷ : IsLocalization.AtPrime Rₚ p
      inst✝⁶ : IsLocalRing Rₚ
      inst✝⁵ : Algebra S Sₚ
      inst✝⁴ : Algebra R Sₚ
      inst✝³ : Algebra Rₚ Sₚ
      inst✝² : IsLocalization (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ
      inst✝¹ : IsScalarTower R S Sₚ
      inst✝ : IsScalarTower R Rₚ Sₚ
      x✝ : S
      this : IsScalarTower R (HasQuotient.Quotient Rₚ (IsLocalRing.maximalIdeal Rₚ)) …
      x : R
      ⊢ Eq ((algebraMap (HasQuotient.Quotient Rₚ (IsLocalRing.maximalIdeal Rₚ)) (Has …
    -/
    rw [← IsScalarTower.algebraMap_apply, ← IsScalarTower.algebraMap_apply]
    /-
      🎉 no goals
    -/


open nonZeroDivisors in
/-- The trace map on `B → A` coincides with the trace map on `B⧸pB → A⧸p`. -/
lemma Algebra.trace_quotient_eq_of_isDedekindDomain (x) [IsDedekindDomain R] [IsDomain S]
    [NoZeroSMulDivisors R S] [Module.Finite R S] [IsIntegrallyClosed S] :
    Algebra.trace (R ⧸ p) (S ⧸ pS) (Ideal.Quotient.mk pS x) =
      Ideal.Quotient.mk p (Algebra.intTrace R S x) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    p : Ideal R
    inst✝⁵ : p.IsMaximal
    x : S
    inst✝⁴ : IsDedekindDomain R
    inst✝³ : IsDomain S
    inst✝² : NoZeroSMulDivisors R S
    inst✝¹ : Module.Finite R S
    inst✝ : IsIntegrallyClosed S
    ⊢ Eq ((Algebra.trace (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal …
  -/
  let Rₚ := Localization.AtPrime p
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    p : Ideal R
    inst✝⁵ : p.IsMaximal
    x : S
    inst✝⁴ : IsDedekindDomain R
    inst✝³ : IsDomain S
    inst✝² : NoZeroSMulDivisors R S
    inst✝¹ : Module.Finite R S
    inst✝ : IsIntegrallyClosed S
    Rₚ : Type u_1 := Localization.AtPrime p
    ⊢ Eq ((Algebra.trace (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal …
  -/
  let Sₚ := Localization (Algebra.algebraMapSubmonoid S p.primeCompl)
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    p : Ideal R
    inst✝⁵ : p.IsMaximal
    x : S
    inst✝⁴ : IsDedekindDomain R
    inst✝³ : IsDomain S
    inst✝² : NoZeroSMulDivisors R S
    inst✝¹ : Module.Finite R S
    inst✝ : IsIntegrallyClosed S
    Rₚ : Type u_1 := Localization.AtPrime p
    Sₚ : Type u_2 := Localization (Algebra.algebraMapSubmonoid S p.primeCompl)
    ⊢ Eq ((Algebra.trace (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal …
  -/
  letI : Algebra Rₚ Sₚ := localizationAlgebra p.primeCompl S
  haveI : IsScalarTower R Rₚ Sₚ := IsScalarTower.of_algebraMap_eq'
    (by rw [RingHom.algebraMap_toAlgebra, IsLocalization.map_comp, ← IsScalarTower.algebraMap_eq])
  haveI : IsLocalization (Submonoid.map (algebraMap R S) (Ideal.primeCompl p)) Sₚ :=
    inferInstanceAs (IsLocalization (Algebra.algebraMapSubmonoid S p.primeCompl) Sₚ)
  have e : Algebra.algebraMapSubmonoid S p.primeCompl ≤ S⁰ :=
    Submonoid.map_le_of_le_comap _ <| p.primeCompl_le_nonZeroDivisors.trans
      (nonZeroDivisors_le_comap_nonZeroDivisors_of_injective _
        (NoZeroSMulDivisors.algebraMap_injective _ _))
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    p : Ideal R
    inst✝⁵ : p.IsMaximal
    x : S
    inst✝⁴ : IsDedekindDomain R
    inst✝³ : IsDomain S
    inst✝² : NoZeroSMulDivisors R S
    inst✝¹ : Module.Finite R S
    inst✝ : IsIntegrallyClosed S
    Rₚ : Type u_1 := Localization.AtPrime p
    Sₚ : Type u_2 := Localization (Algebra.algebraMapSubmonoid S p.primeCompl)
    this✝¹ : Algebra Rₚ Sₚ := localizationAlgebra p.primeCompl S
    this✝ : IsScalarTower R Rₚ Sₚ
    this : IsLocalization (Submonoid.map (algebraMap R S) p.primeCompl) Sₚ
    e : LE.le (Algebra.algebraMapSubmonoid S p.primeCompl) (nonZeroDivisors S)
    ⊢ Eq ((Algebra.trace (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal …
  -/
  haveI : IsDomain Sₚ := IsLocalization.isDomain_of_le_nonZeroDivisors S e
  haveI : NoZeroSMulDivisors Rₚ Sₚ := by
    rw [NoZeroSMulDivisors.iff_algebraMap_injective, RingHom.injective_iff_ker_eq_bot,
      RingHom.ker_eq_bot_iff_eq_zero]
    intro x hx
    obtain ⟨x, s, rfl⟩ := IsLocalization.mk'_surjective p.primeCompl x
    simp only [Sₚ, RingHom.algebraMap_toAlgebra, IsLocalization.map_mk',
      IsLocalization.mk'_eq_zero_iff, mul_eq_zero, Subtype.exists, exists_prop] at hx ⊢
    obtain ⟨_, ⟨a, ha, rfl⟩, H⟩ := hx
    simp only [(injective_iff_map_eq_zero' _).mp (NoZeroSMulDivisors.algebraMap_injective R S)] at H
    refine ⟨a, ha, H⟩
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    p : Ideal R
    inst✝⁵ : p.IsMaximal
    x : S
    inst✝⁴ : IsDedekindDomain R
    inst✝³ : IsDomain S
    inst✝² : NoZeroSMulDivisors R S
    inst✝¹ : Module.Finite R S
    inst✝ : IsIntegrallyClosed S
    Rₚ : Type u_1 := Localization.AtPrime p
    Sₚ : Type u_2 := Localization (Algebra.algebraMapSubmonoid S p.primeCompl)
    this✝³ : Algebra Rₚ Sₚ := localizationAlgebra p.primeCompl S
    this✝² : IsScalarTower R Rₚ Sₚ
    this✝¹ : IsLocalization (Submonoid.map (algebraMap R S) p.primeCompl) Sₚ
    e : LE.le (Algebra.algebraMapSubmonoid S p.primeCompl) (nonZeroDivisors S)
    this✝ : IsDomain Sₚ
    this : NoZeroSMulDivisors Rₚ Sₚ
    ⊢ Eq ((Algebra.trace (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal …
  -/
  haveI : Module.Finite Rₚ Sₚ := Module.Finite_of_isLocalization R S _ _ p.primeCompl
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    p : Ideal R
    inst✝⁵ : p.IsMaximal
    x : S
    inst✝⁴ : IsDedekindDomain R
    inst✝³ : IsDomain S
    inst✝² : NoZeroSMulDivisors R S
    inst✝¹ : Module.Finite R S
    inst✝ : IsIntegrallyClosed S
    Rₚ : Type u_1 := Localization.AtPrime p
    Sₚ : Type u_2 := Localization (Algebra.algebraMapSubmonoid S p.primeCompl)
    this✝⁴ : Algebra Rₚ Sₚ := localizationAlgebra p.primeCompl S
    this✝³ : IsScalarTower R Rₚ Sₚ
    this✝² : IsLocalization (Submonoid.map (algebraMap R S) p.primeCompl) Sₚ
    e : LE.le (Algebra.algebraMapSubmonoid S p.primeCompl) (nonZeroDivisors S)
    this✝¹ : IsDomain Sₚ
    this✝ : NoZeroSMulDivisors Rₚ Sₚ
    this : Module.Finite Rₚ Sₚ
    ⊢ Eq ((Algebra.trace (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal …
  -/
  haveI : IsIntegrallyClosed Sₚ := isIntegrallyClosed_of_isLocalization _ _ e
  have : IsPrincipalIdealRing Rₚ := by
    by_cases hp : p = ⊥
    · infer_instance
    · have := (IsDedekindDomain.isDedekindDomainDvr R).2 p hp inferInstance
      infer_instance
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    p : Ideal R
    inst✝⁵ : p.IsMaximal
    x : S
    inst✝⁴ : IsDedekindDomain R
    inst✝³ : IsDomain S
    inst✝² : NoZeroSMulDivisors R S
    inst✝¹ : Module.Finite R S
    inst✝ : IsIntegrallyClosed S
    Rₚ : Type u_1 := Localization.AtPrime p
    Sₚ : Type u_2 := Localization (Algebra.algebraMapSubmonoid S p.primeCompl)
    this✝⁶ : Algebra Rₚ Sₚ := localizationAlgebra p.primeCompl S
    this✝⁵ : IsScalarTower R Rₚ Sₚ
    this✝⁴ : IsLocalization (Submonoid.map (algebraMap R S) p.primeCompl) Sₚ
    e : LE.le (Algebra.algebraMapSubmonoid S p.primeCompl) (nonZeroDivisors S)
    this✝³ : IsDomain Sₚ
    this✝² : NoZeroSMulDivisors Rₚ Sₚ
    this✝¹ : Module.Finite Rₚ Sₚ
    this✝ : IsIntegrallyClosed Sₚ
    this : IsPrincipalIdealRing Rₚ
    ⊢ Eq ((Algebra.trace (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal …
  -/
  haveI : Module.Free Rₚ Sₚ := Module.free_of_finite_type_torsion_free'
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    p : Ideal R
    inst✝⁵ : p.IsMaximal
    x : S
    inst✝⁴ : IsDedekindDomain R
    inst✝³ : IsDomain S
    inst✝² : NoZeroSMulDivisors R S
    inst✝¹ : Module.Finite R S
    inst✝ : IsIntegrallyClosed S
    Rₚ : Type u_1 := Localization.AtPrime p
    Sₚ : Type u_2 := Localization (Algebra.algebraMapSubmonoid S p.primeCompl)
    this✝⁷ : Algebra Rₚ Sₚ := localizationAlgebra p.primeCompl S
    this✝⁶ : IsScalarTower R Rₚ Sₚ
    this✝⁵ : IsLocalization (Submonoid.map (algebraMap R S) p.primeCompl) Sₚ
    e : LE.le (Algebra.algebraMapSubmonoid S p.primeCompl) (nonZeroDivisors S)
    this✝⁴ : IsDomain Sₚ
    this✝³ : NoZeroSMulDivisors Rₚ Sₚ
    this✝² : Module.Finite Rₚ Sₚ
    this✝¹ : IsIntegrallyClosed Sₚ
    this✝ : IsPrincipalIdealRing Rₚ
    this : Module.Free Rₚ Sₚ
    ⊢ Eq ((Algebra.trace (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal …
  -/
  apply (equivQuotMaximalIdealOfIsLocalization p Rₚ).injective
  rw [trace_quotient_eq_trace_localization_quotient S p Rₚ Sₚ, IsScalarTower.algebraMap_eq S Sₚ,
    RingHom.comp_apply, Ideal.Quotient.algebraMap_eq, Algebra.trace_quotient_mk,
    RingEquiv.apply_symm_apply, ← Algebra.intTrace_eq_trace,
    ← Algebra.intTrace_eq_of_isLocalization R S p.primeCompl (Aₘ := Rₚ) (Bₘ := Sₚ) x,
    ← Ideal.Quotient.algebraMap_eq, ← IsScalarTower.algebraMap_apply]
  simp only [equivQuotMaximalIdealOfIsLocalization, RingHom.quotientKerEquivOfSurjective,
    RingEquiv.coe_trans, Function.comp_apply, Ideal.quotEquivOfEq_mk,
    RingHom.quotientKerEquivOfRightInverse.apply, RingHom.kerLift_mk]


