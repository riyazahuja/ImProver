/--
If `S` is the localization of `R` at a submonoid `M`, then `MvPolynomial σ S`
is the localization of `MvPolynomial σ R` at `M.map MvPolynomial.C`.
-/
instance isLocalization : IsLocalization (M.map <| C (σ := σ))
    (MvPolynomial σ S) where
  map_units' := by
    /-
      σ : Type u_1
      R : Type u_2
      inst✝³ : CommRing R
      M : Submonoid R
      S : Type u_3
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      ⊢ ∀ (y : Subtype fun x => Membership.mem (Submonoid.map MvPolynomial.C M) x),  …
    -/
    rintro ⟨p, q, hq, rfl⟩
    /-
      case mk.intro.intro
      σ : Type u_1
      R : Type u_2
      inst✝³ : CommRing R
      M : Submonoid R
      S : Type u_3
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      q : R
      hq : Membership.mem (↑M) q
      ⊢ IsUnit ((algebraMap (MvPolynomial σ R) (MvPolynomial σ S)) ↑⟨MvPolynomial.C  …
    -/
    simp only [algebraMap_def, map_C]
    /-
      case mk.intro.intro
      σ : Type u_1
      R : Type u_2
      inst✝³ : CommRing R
      M : Submonoid R
      S : Type u_3
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      q : R
      hq : Membership.mem (↑M) q
      ⊢ IsUnit (MvPolynomial.C ((algebraMap R S) q))
    -/
    exact IsUnit.map _ (IsLocalization.map_units _ ⟨q, hq⟩)
    /-
      🎉 no goals
    -/
  surj' p := by
    simp only [algebraMap_def, Prod.exists, Subtype.exists,
      Submonoid.mem_map, exists_prop, exists_exists_and_eq_and, map_C]
    /-
      σ : Type u_1
      R : Type u_2
      inst✝³ : CommRing R
      M : Submonoid R
      S : Type u_3
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      p : MvPolynomial σ S
      ⊢ Exists fun a => Exists fun a_1 => And (Membership.mem M a_1) (Eq (HMul.hMul  …
    -/
    refine induction_on' p ?_ ?_
      /-
        case refine_1
        σ : Type u_1
        R : Type u_2
        inst✝³ : CommRing R
        M : Submonoid R
        S : Type u_3
        inst✝² : CommRing S
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization M S
        p : MvPolynomial σ S
        ⊢ ∀ (u : Finsupp σ Nat) (a : S), Exists fun a_1 => Exists fun a_2 => And (Memb …
      -/
    · intro u s
      /-
        case refine_1
        σ : Type u_1
        R : Type u_2
        inst✝³ : CommRing R
        M : Submonoid R
        S : Type u_3
        inst✝² : CommRing S
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization M S
        p : MvPolynomial σ S
        u : Finsupp σ Nat
        s : S
        ⊢ Exists fun a => Exists fun a_1 => And (Membership.mem M a_1) (Eq (HMul.hMul  …
      -/
      obtain ⟨⟨r, m⟩, hr⟩ := IsLocalization.surj M s
      /-
        case refine_1.intro.mk
        σ : Type u_1
        R : Type u_2
        inst✝³ : CommRing R
        M : Submonoid R
        S : Type u_3
        inst✝² : CommRing S
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization M S
        p : MvPolynomial σ S
        u : Finsupp σ Nat
        s : S
        r : R
        m : Subtype fun x => Membership.mem M x
        hr : Eq (HMul.hMul s ((algebraMap R S) ↑{ fst := r, snd := m }.2)) ((algebraMa …
        ⊢ Exists fun a => Exists fun a_1 => And (Membership.mem M a_1) (Eq (HMul.hMul  …
      -/
      use monomial u r, m, m.property
      /-
        case right
        σ : Type u_1
        R : Type u_2
        inst✝³ : CommRing R
        M : Submonoid R
        S : Type u_3
        inst✝² : CommRing S
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization M S
        p : MvPolynomial σ S
        u : Finsupp σ Nat
        s : S
        r : R
        m : Subtype fun x => Membership.mem M x
        hr : Eq (HMul.hMul s ((algebraMap R S) ↑{ fst := r, snd := m }.2)) ((algebraMa …
        ⊢ Eq (HMul.hMul ((MvPolynomial.monomial u) s) (MvPolynomial.C ((algebraMap R S …
      -/
      simp only [map_monomial]
      /-
        case right
        σ : Type u_1
        R : Type u_2
        inst✝³ : CommRing R
        M : Submonoid R
        S : Type u_3
        inst✝² : CommRing S
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization M S
        p : MvPolynomial σ S
        u : Finsupp σ Nat
        s : S
        r : R
        m : Subtype fun x => Membership.mem M x
        hr : Eq (HMul.hMul s ((algebraMap R S) ↑{ fst := r, snd := m }.2)) ((algebraMa …
        ⊢ Eq (HMul.hMul ((MvPolynomial.monomial u) s) (MvPolynomial.C ((algebraMap R S …
      -/
      rw [← hr, mul_comm, C_mul_monomial, mul_comm]
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        σ : Type u_1
        R : Type u_2
        inst✝³ : CommRing R
        M : Submonoid R
        S : Type u_3
        inst✝² : CommRing S
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization M S
        p : MvPolynomial σ S
        ⊢ ∀ (p q : MvPolynomial σ S), (Exists fun a => Exists fun a_1 => And (Membersh …
      -/
    · intro p p' ⟨x, m, hm, hxm⟩ ⟨x', m', hm', hxm'⟩
      /-
        case refine_2
        σ : Type u_1
        R : Type u_2
        inst✝³ : CommRing R
        M : Submonoid R
        S : Type u_3
        inst✝² : CommRing S
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization M S
        p✝ p p' : MvPolynomial σ S
        x : MvPolynomial σ R
        m : R
        hm : Membership.mem M m
        hxm : Eq (HMul.hMul p (MvPolynomial.C ((algebraMap R S) m))) ((MvPolynomial.ma …
        x' : MvPolynomial σ R
        m' : R
        hm' : Membership.mem M m'
        hxm' : Eq (HMul.hMul p' (MvPolynomial.C ((algebraMap R S) m'))) ((MvPolynomial …
        ⊢ Exists fun a => Exists fun a_1 => And (Membership.mem M a_1) (Eq (HMul.hMul  …
      -/
      use x * (C m') + x' * (C m), m * m', Submonoid.mul_mem _ hm hm'
      /-
        case right
        σ : Type u_1
        R : Type u_2
        inst✝³ : CommRing R
        M : Submonoid R
        S : Type u_3
        inst✝² : CommRing S
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization M S
        p✝ p p' : MvPolynomial σ S
        x : MvPolynomial σ R
        m : R
        hm : Membership.mem M m
        hxm : Eq (HMul.hMul p (MvPolynomial.C ((algebraMap R S) m))) ((MvPolynomial.ma …
        x' : MvPolynomial σ R
        m' : R
        hm' : Membership.mem M m'
        hxm' : Eq (HMul.hMul p' (MvPolynomial.C ((algebraMap R S) m'))) ((MvPolynomial …
        ⊢ Eq (HMul.hMul (HAdd.hAdd p p') (MvPolynomial.C ((algebraMap R S) (HMul.hMul  …
      -/
      simp only [map_mul, map_add, map_C]
      /-
        case right
        σ : Type u_1
        R : Type u_2
        inst✝³ : CommRing R
        M : Submonoid R
        S : Type u_3
        inst✝² : CommRing S
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization M S
        p✝ p p' : MvPolynomial σ S
        x : MvPolynomial σ R
        m : R
        hm : Membership.mem M m
        hxm : Eq (HMul.hMul p (MvPolynomial.C ((algebraMap R S) m))) ((MvPolynomial.ma …
        x' : MvPolynomial σ R
        m' : R
        hm' : Membership.mem M m'
        hxm' : Eq (HMul.hMul p' (MvPolynomial.C ((algebraMap R S) m'))) ((MvPolynomial …
        ⊢ Eq (HMul.hMul (HAdd.hAdd p p') (HMul.hMul (MvPolynomial.C ((algebraMap R S)  …
      -/
      rw [add_mul, ← mul_assoc, hxm, ← mul_assoc, ← hxm, ← hxm']
      /-
        case right
        σ : Type u_1
        R : Type u_2
        inst✝³ : CommRing R
        M : Submonoid R
        S : Type u_3
        inst✝² : CommRing S
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization M S
        p✝ p p' : MvPolynomial σ S
        x : MvPolynomial σ R
        m : R
        hm : Membership.mem M m
        hxm : Eq (HMul.hMul p (MvPolynomial.C ((algebraMap R S) m))) ((MvPolynomial.ma …
        x' : MvPolynomial σ R
        m' : R
        hm' : Membership.mem M m'
        hxm' : Eq (HMul.hMul p' (MvPolynomial.C ((algebraMap R S) m'))) ((MvPolynomial …
        ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul p (MvPolynomial.C ((algebraMap R S) m))) …
      -/
      ring
      /-
        🎉 no goals
      -/
  exists_of_eq {p q} := by
    /-
      σ : Type u_1
      R : Type u_2
      inst✝³ : CommRing R
      M : Submonoid R
      S : Type u_3
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      p q : MvPolynomial σ R
      ⊢ Eq ((algebraMap (MvPolynomial σ R) (MvPolynomial σ S)) p) ((algebraMap (MvPo …
    -/
    intro h
    /-
      σ : Type u_1
      R : Type u_2
      inst✝³ : CommRing R
      M : Submonoid R
      S : Type u_3
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      p q : MvPolynomial σ R
      h : Eq ((algebraMap (MvPolynomial σ R) (MvPolynomial σ S)) p) ((algebraMap (Mv …
      ⊢ Exists fun c => Eq (HMul.hMul (↑c) p) (HMul.hMul (↑c) q)
    -/
    simp_rw [algebraMap_def, MvPolynomial.ext_iff, coeff_map] at h
    /-
      σ : Type u_1
      R : Type u_2
      inst✝³ : CommRing R
      M : Submonoid R
      S : Type u_3
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      p q : MvPolynomial σ R
      h : ∀ (m : Finsupp σ Nat), Eq ((algebraMap R S) (MvPolynomial.coeff m p)) ((al …
      ⊢ Exists fun c => Eq (HMul.hMul (↑c) p) (HMul.hMul (↑c) q)
    -/
    choose c hc using (fun m ↦ IsLocalization.exists_of_eq (M := M) (h m))
    /-
      σ : Type u_1
      R : Type u_2
      inst✝³ : CommRing R
      M : Submonoid R
      S : Type u_3
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      p q : MvPolynomial σ R
      h : ∀ (m : Finsupp σ Nat), Eq ((algebraMap R S) (MvPolynomial.coeff m p)) ((al …
      c : Finsupp σ Nat → Subtype fun x => Membership.mem M x
      hc : ∀ (m : Finsupp σ Nat), Eq (HMul.hMul (↑(c m)) (MvPolynomial.coeff m p)) ( …
      ⊢ Exists fun c => Eq (HMul.hMul (↑c) p) (HMul.hMul (↑c) q)
    -/
    simp only [Subtype.exists, Submonoid.mem_map, exists_prop, exists_exists_and_eq_and]
    classical
    refine ⟨Finset.prod (p.support ∪ q.support) (fun m ↦ c m), ?_, ?_⟩
    · exact M.prod_mem (fun m _ ↦ (c m).property)
    · ext m
      simp only [coeff_C_mul]
      by_cases h : m ∈ p.support ∪ q.support
      · exact Finset.prod_mul_eq_prod_mul_of_exists m h (hc m)
      · simp only [Finset.mem_union, mem_support_iff, ne_eq, not_or, Decidable.not_not] at h
        rw [h.left, h.right]


lemma isLocalization_C_mk' (a : R) (m : M) :
    C (IsLocalization.mk' S a m) = IsLocalization.mk' (MvPolynomial σ S) (C (σ := σ) a)
      ⟨C m, Submonoid.mem_map_of_mem C m.property⟩ := by
  simp_rw [IsLocalization.eq_mk'_iff_mul_eq, algebraMap_def, map_C, ← map_mul,
    IsLocalization.mk'_spec]


/-- The canonical algebra map from `MvPolynomial Unit R` quotiented by
`C r * X () - 1` to the localization of `R` away from `r`. -/
private noncomputable
def auxHom : (MvPolynomial Unit R) ⧸ (Ideal.span { C r * X () - 1 }) →ₐ[R] S :=
  Ideal.Quotient.liftₐ (Ideal.span { C r * X () - 1}) (aeval (fun _ ↦ invSelf r)) <| by
    /-
      σ : Type u_1
      R : Type u_2
      inst✝³ : CommRing R
      M : Submonoid R
      S : Type u_3
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      r : R
      inst✝ : IsLocalization.Away r S
      ⊢ ∀ (a : MvPolynomial Unit R), Membership.mem (Ideal.span (Singleton.singleton …
    -/
    intro p hp
    /-
      σ : Type u_1
      R : Type u_2
      inst✝³ : CommRing R
      M : Submonoid R
      S : Type u_3
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      r : R
      inst✝ : IsLocalization.Away r S
      p : MvPolynomial Unit R
      hp : Membership.mem (Ideal.span (Singleton.singleton (HSub.hSub (HMul.hMul (Mv …
      ⊢ Eq ((MvPolynomial.aeval fun x => IsLocalization.Away.invSelf r) p) 0
    -/
    refine Submodule.span_induction ?_ ?_ ?_ ?_ hp
      /-
        case refine_1
        σ : Type u_1
        R : Type u_2
        inst✝³ : CommRing R
        M : Submonoid R
        S : Type u_3
        inst✝² : CommRing S
        inst✝¹ : Algebra R S
        r : R
        inst✝ : IsLocalization.Away r S
        p : MvPolynomial Unit R
        hp : Membership.mem (Ideal.span (Singleton.singleton (HSub.hSub (HMul.hMul (Mv …
        ⊢ ∀ (x : MvPolynomial Unit R), Membership.mem (Singleton.singleton (HSub.hSub  …
      -/
    · rintro p ⟨q, rfl⟩
      /-
        case refine_1.refl
        σ : Type u_1
        R : Type u_2
        inst✝³ : CommRing R
        M : Submonoid R
        S : Type u_3
        inst✝² : CommRing S
        inst✝¹ : Algebra R S
        r : R
        inst✝ : IsLocalization.Away r S
        p : MvPolynomial Unit R
        hp : Membership.mem (Ideal.span (Singleton.singleton (HSub.hSub (HMul.hMul (Mv …
        ⊢ Eq ((MvPolynomial.aeval fun x => IsLocalization.Away.invSelf r) (HSub.hSub ( …
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        σ : Type u_1
        R : Type u_2
        inst✝³ : CommRing R
        M : Submonoid R
        S : Type u_3
        inst✝² : CommRing S
        inst✝¹ : Algebra R S
        r : R
        inst✝ : IsLocalization.Away r S
        p : MvPolynomial Unit R
        hp : Membership.mem (Ideal.span (Singleton.singleton (HSub.hSub (HMul.hMul (Mv …
        ⊢ Eq ((MvPolynomial.aeval fun x => IsLocalization.Away.invSelf r) 0) 0
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case refine_3
        σ : Type u_1
        R : Type u_2
        inst✝³ : CommRing R
        M : Submonoid R
        S : Type u_3
        inst✝² : CommRing S
        inst✝¹ : Algebra R S
        r : R
        inst✝ : IsLocalization.Away r S
        p : MvPolynomial Unit R
        hp : Membership.mem (Ideal.span (Singleton.singleton (HSub.hSub (HMul.hMul (Mv …
        ⊢ ∀ (x y : MvPolynomial Unit R), Membership.mem (Submodule.span (MvPolynomial  …
      -/
    · intro p q _ _ hp hq
      /-
        case refine_3
        σ : Type u_1
        R : Type u_2
        inst✝³ : CommRing R
        M : Submonoid R
        S : Type u_3
        inst✝² : CommRing S
        inst✝¹ : Algebra R S
        r : R
        inst✝ : IsLocalization.Away r S
        p✝ : MvPolynomial Unit R
        hp✝ : Membership.mem (Ideal.span (Singleton.singleton (HSub.hSub (HMul.hMul (M …
        p q : MvPolynomial Unit R
        hx✝ : Membership.mem (Submodule.span (MvPolynomial Unit R) (Singleton.singleto …
        hy✝ : Membership.mem (Submodule.span (MvPolynomial Unit R) (Singleton.singleto …
        hp : Eq ((MvPolynomial.aeval fun x => IsLocalization.Away.invSelf r) p) 0
        hq : Eq ((MvPolynomial.aeval fun x => IsLocalization.Away.invSelf r) q) 0
        ⊢ Eq ((MvPolynomial.aeval fun x => IsLocalization.Away.invSelf r) (HAdd.hAdd p …
      -/
      simp [hp, hq]
      /-
        🎉 no goals
      -/
      /-
        case refine_4
        σ : Type u_1
        R : Type u_2
        inst✝³ : CommRing R
        M : Submonoid R
        S : Type u_3
        inst✝² : CommRing S
        inst✝¹ : Algebra R S
        r : R
        inst✝ : IsLocalization.Away r S
        p : MvPolynomial Unit R
        hp : Membership.mem (Ideal.span (Singleton.singleton (HSub.hSub (HMul.hMul (Mv …
        ⊢ ∀ (a x : MvPolynomial Unit R), Membership.mem (Submodule.span (MvPolynomial  …
      -/
    · intro a x _ hx
      /-
        case refine_4
        σ : Type u_1
        R : Type u_2
        inst✝³ : CommRing R
        M : Submonoid R
        S : Type u_3
        inst✝² : CommRing S
        inst✝¹ : Algebra R S
        r : R
        inst✝ : IsLocalization.Away r S
        p : MvPolynomial Unit R
        hp : Membership.mem (Ideal.span (Singleton.singleton (HSub.hSub (HMul.hMul (Mv …
        a x : MvPolynomial Unit R
        hx✝ : Membership.mem (Submodule.span (MvPolynomial Unit R) (Singleton.singleto …
        hx : Eq ((MvPolynomial.aeval fun x => IsLocalization.Away.invSelf r) x) 0
        ⊢ Eq ((MvPolynomial.aeval fun x => IsLocalization.Away.invSelf r) (HSMul.hSMul …
      -/
      simp [hx]
      /-
        🎉 no goals
      -/


@[simp]
private lemma auxHom_mk (p : MvPolynomial Unit R) :
    auxHom S r p = aeval (S₁ := S) (fun _ ↦ invSelf r) p :=
  rfl


private noncomputable
def auxInv : S →+* (MvPolynomial Unit R) ⧸ Ideal.span { C r * X () - 1 } :=
  letI g : R →+* MvPolynomial Unit R ⧸ (Ideal.span { C r * X () - 1 }) :=
    (Ideal.Quotient.mk _).comp C
  IsLocalization.Away.lift (S := S) (g := g) r <| by
    /-
      σ : Type u_1
      R : Type u_2
      inst✝³ : CommRing R
      M : Submonoid R
      S : Type u_3
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      r : R
      inst✝ : IsLocalization.Away r S
      g : RingHom R (HasQuotient.Quotient (MvPolynomial Unit R) (Ideal.span (Singlet …
      ⊢ IsUnit (g r)
    -/
    simp only [RingHom.coe_comp, Function.comp_apply, g]
    /-
      σ : Type u_1
      R : Type u_2
      inst✝³ : CommRing R
      M : Submonoid R
      S : Type u_3
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      r : R
      inst✝ : IsLocalization.Away r S
      g : RingHom R (HasQuotient.Quotient (MvPolynomial Unit R) (Ideal.span (Singlet …
      ⊢ IsUnit ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton (HSub.hSub (HMul …
    -/
    rw [isUnit_iff_exists_inv]
    /-
      σ : Type u_1
      R : Type u_2
      inst✝³ : CommRing R
      M : Submonoid R
      S : Type u_3
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      r : R
      inst✝ : IsLocalization.Away r S
      g : RingHom R (HasQuotient.Quotient (MvPolynomial Unit R) (Ideal.span (Singlet …
      ⊢ Exists fun b => Eq (HMul.hMul ((Ideal.Quotient.mk (Ideal.span (Singleton.sin …
    -/
    use (Ideal.Quotient.mk _ <| X ())
    /-
      case h
      σ : Type u_1
      R : Type u_2
      inst✝³ : CommRing R
      M : Submonoid R
      S : Type u_3
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      r : R
      inst✝ : IsLocalization.Away r S
      g : RingHom R (HasQuotient.Quotient (MvPolynomial Unit R) (Ideal.span (Singlet …
      ⊢ Eq (HMul.hMul ((Ideal.Quotient.mk (Ideal.span (Singleton.singleton (HSub.hSu …
    -/
    rw [← map_mul, ← map_one (Ideal.Quotient.mk _), Ideal.Quotient.mk_eq_mk_iff_sub_mem]
    /-
      case h
      σ : Type u_1
      R : Type u_2
      inst✝³ : CommRing R
      M : Submonoid R
      S : Type u_3
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      r : R
      inst✝ : IsLocalization.Away r S
      g : RingHom R (HasQuotient.Quotient (MvPolynomial Unit R) (Ideal.span (Singlet …
      ⊢ Membership.mem (Ideal.span (Singleton.singleton (HSub.hSub (HMul.hMul (MvPol …
    -/
    exact Ideal.mem_span_singleton_self (C r * X () - 1)
    /-
      🎉 no goals
    -/


private lemma auxHom_auxInv : (auxHom S r).toRingHom.comp (auxInv S r) = RingHom.id S := by
  /-
    R : Type u_2
    inst✝³ : CommRing R
    S : Type u_3
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    r : R
    inst✝ : IsLocalization.Away r S
    ⊢ Eq ((IsLocalization.Away.auxHom S r).comp (IsLocalization.Away.auxInv S r))  …
  -/
  apply IsLocalization.ringHom_ext (Submonoid.powers r)
  /-
    case h
    R : Type u_2
    inst✝³ : CommRing R
    S : Type u_3
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    r : R
    inst✝ : IsLocalization.Away r S
    ⊢ Eq (((IsLocalization.Away.auxHom S r).comp (IsLocalization.Away.auxInv S r)) …
  -/
  ext x
  /-
    case h.a
    R : Type u_2
    inst✝³ : CommRing R
    S : Type u_3
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    r : R
    inst✝ : IsLocalization.Away r S
    x : R
    ⊢ Eq ((((IsLocalization.Away.auxHom S r).comp (IsLocalization.Away.auxInv S r) …
  -/
  simp [auxInv]
  /-
    🎉 no goals
  -/


private lemma auxInv_auxHom : (auxInv S r).comp (auxHom (S := S) r).toRingHom = RingHom.id _ := by
  /-
    R : Type u_2
    inst✝³ : CommRing R
    S : Type u_3
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    r : R
    inst✝ : IsLocalization.Away r S
    ⊢ Eq ((IsLocalization.Away.auxInv S r).comp (IsLocalization.Away.auxHom S r).t …
  -/
  rw [← RingHom.cancel_right (Ideal.Quotient.mk_surjective)]
  /-
    R : Type u_2
    inst✝³ : CommRing R
    S : Type u_3
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    r : R
    inst✝ : IsLocalization.Away r S
    ⊢ Eq (((IsLocalization.Away.auxInv S r).comp (IsLocalization.Away.auxHom S r). …
  -/
  ext x
    /-
      case hC.a
      R : Type u_2
      inst✝³ : CommRing R
      S : Type u_3
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      r : R
      inst✝ : IsLocalization.Away r S
      x : R
      ⊢ Eq (((((IsLocalization.Away.auxInv S r).comp (IsLocalization.Away.auxHom S r …
    -/
  · simp [auxInv]
    /-
      🎉 no goals
    -/
  · simp only [auxInv, AlgHom.toRingHom_eq_coe, RingHom.coe_comp, RingHom.coe_coe,
      Function.comp_apply, auxHom_mk, aeval_X, RingHomCompTriple.comp_eq]
    /-
      case hX
      R : Type u_2
      inst✝³ : CommRing R
      S : Type u_3
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      r : R
      inst✝ : IsLocalization.Away r S
      x : Unit
      ⊢ Eq ((IsLocalization.Away.lift r ⋯) (IsLocalization.Away.invSelf r)) ((Ideal. …
    -/
    erw [IsLocalization.lift_mk'_spec]
    /-
      case hX
      R : Type u_2
      inst✝³ : CommRing R
      S : Type u_3
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      r : R
      inst✝ : IsLocalization.Away r S
      x : Unit
      ⊢ Eq (((Ideal.Quotient.mk (Ideal.span (Singleton.singleton (HSub.hSub (HMul.hM …
    -/
    simp only [map_one, RingHom.coe_comp, Function.comp_apply]
    rw [← map_one (Ideal.Quotient.mk _), ← map_mul, Ideal.Quotient.mk_eq_mk_iff_sub_mem,
      ← Ideal.neg_mem_iff, neg_sub]
    /-
      case hX
      R : Type u_2
      inst✝³ : CommRing R
      S : Type u_3
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      r : R
      inst✝ : IsLocalization.Away r S
      x : Unit
      ⊢ Membership.mem (Ideal.span (Singleton.singleton (HSub.hSub (HMul.hMul (MvPol …
    -/
    exact Ideal.mem_span_singleton_self (C r * X x - 1)
    /-
      🎉 no goals
    -/


/-- The canonical algebra isomorphism from `MvPolynomial Unit R` quotiented by
`C r * X () - 1` to the localization of `R` away from `r`. -/
noncomputable def mvPolynomialQuotientEquiv :
    ((MvPolynomial Unit R) ⧸ Ideal.span { C r * X () - 1 }) ≃ₐ[R] S where
  toFun := auxHom S r
  invFun := auxInv S r
  left_inv x := by
    /-
      σ : Type u_1
      R : Type u_2
      inst✝³ : CommRing R
      M : Submonoid R
      S : Type u_3
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      r : R
      inst✝ : IsLocalization.Away r S
      x : HasQuotient.Quotient (MvPolynomial Unit R) (Ideal.span (Singleton.singleto …
      ⊢ Eq ((IsLocalization.Away.auxInv S r) ((IsLocalization.Away.auxHom S r) x)) x
    -/
    simpa using congrFun (congrArg DFunLike.coe <| auxInv_auxHom S r) x
    /-
      🎉 no goals
    -/
  right_inv s := by
    /-
      σ : Type u_1
      R : Type u_2
      inst✝³ : CommRing R
      M : Submonoid R
      S : Type u_3
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      r : R
      inst✝ : IsLocalization.Away r S
      s : S
      ⊢ Eq ((IsLocalization.Away.auxHom S r) ((IsLocalization.Away.auxInv S r) s)) s
    -/
    simpa using congrFun (congrArg DFunLike.coe <| auxHom_auxInv S r) s
    /-
      🎉 no goals
    -/
                 /-
                   σ : Type u_1
                   R : Type u_2
                   inst✝³ : CommRing R
                   M : Submonoid R
                   S : Type u_3
                   inst✝² : CommRing S
                   inst✝¹ : Algebra R S
                   r : R
                   inst✝ : IsLocalization.Away r S
                   ⊢ ∀ (x y : HasQuotient.Quotient (MvPolynomial Unit R) (Ideal.span (Singleton.s …
                 -/
  map_mul' := by simp
                 /-
                   🎉 no goals
                 -/
                 /-
                   σ : Type u_1
                   R : Type u_2
                   inst✝³ : CommRing R
                   M : Submonoid R
                   S : Type u_3
                   inst✝² : CommRing S
                   inst✝¹ : Algebra R S
                   r : R
                   inst✝ : IsLocalization.Away r S
                   ⊢ ∀ (x y : HasQuotient.Quotient (MvPolynomial Unit R) (Ideal.span (Singleton.s …
                 -/
  map_add' := by simp
                 /-
                   🎉 no goals
                 -/
                  /-
                    σ : Type u_1
                    R : Type u_2
                    inst✝³ : CommRing R
                    M : Submonoid R
                    S : Type u_3
                    inst✝² : CommRing S
                    inst✝¹ : Algebra R S
                    r : R
                    inst✝ : IsLocalization.Away r S
                    ⊢ ∀ (r_1 : R), Eq ({ toFun := ⇑(IsLocalization.Away.auxHom S r), invFun := ⇑(I …
                  -/
  commutes' := by simp
                  /-
                    🎉 no goals
                  -/


@[simp]
lemma mvPolynomialQuotientEquiv_apply (p : MvPolynomial Unit R) :
    mvPolynomialQuotientEquiv S r (Ideal.Quotient.mk _ p) = aeval (S₁ := S) (fun _ ↦ invSelf r) p :=
  rfl


