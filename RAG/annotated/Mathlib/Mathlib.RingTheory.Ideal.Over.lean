theorem coeff_zero_mem_comap_of_root_mem_of_eval_mem {r : S} (hr : r ∈ I) {p : R[X]}
    (hp : p.eval₂ f r ∈ I) : p.coeff 0 ∈ I.comap f := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    S : Type u_2
    inst✝ : CommRing S
    f : RingHom R S
    I : Ideal S
    r : S
    hr : Membership.mem I r
    p : Polynomial R
    hp : Membership.mem I (Polynomial.eval₂ f r p)
    ⊢ Membership.mem (Ideal.comap f I) (p.coeff 0)
  -/
  rw [← p.divX_mul_X_add, eval₂_add, eval₂_C, eval₂_mul, eval₂_X] at hp
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    S : Type u_2
    inst✝ : CommRing S
    f : RingHom R S
    I : Ideal S
    r : S
    hr : Membership.mem I r
    p : Polynomial R
    hp : Membership.mem I (HAdd.hAdd (HMul.hMul (Polynomial.eval₂ f r p.divX) r) ( …
    ⊢ Membership.mem (Ideal.comap f I) (p.coeff 0)
  -/
  refine mem_comap.mpr ((I.add_mem_iff_right ?_).mp hp)
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    S : Type u_2
    inst✝ : CommRing S
    f : RingHom R S
    I : Ideal S
    r : S
    hr : Membership.mem I r
    p : Polynomial R
    hp : Membership.mem I (HAdd.hAdd (HMul.hMul (Polynomial.eval₂ f r p.divX) r) ( …
    ⊢ Membership.mem I (HMul.hMul (Polynomial.eval₂ f r p.divX) r)
  -/
  exact I.mul_mem_left _ hr
  /-
    🎉 no goals
  -/


theorem coeff_zero_mem_comap_of_root_mem {r : S} (hr : r ∈ I) {p : R[X]} (hp : p.eval₂ f r = 0) :
    p.coeff 0 ∈ I.comap f :=
  coeff_zero_mem_comap_of_root_mem_of_eval_mem hr (hp.symm ▸ I.zero_mem)


theorem exists_coeff_ne_zero_mem_comap_of_non_zero_divisor_root_mem {r : S}
    (r_non_zero_divisor : ∀ {x}, x * r = 0 → x = 0) (hr : r ∈ I) {p : R[X]} :
    p ≠ 0 → p.eval₂ f r = 0 → ∃ i, p.coeff i ≠ 0 ∧ p.coeff i ∈ I.comap f := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    S : Type u_2
    inst✝ : CommRing S
    f : RingHom R S
    I : Ideal S
    r : S
    r_non_zero_divisor : ∀ {x : S}, Eq (HMul.hMul x r) 0 → Eq x 0
    hr : Membership.mem I r
    p : Polynomial R
    ⊢ Ne p 0 → Eq (Polynomial.eval₂ f r p) 0 → Exists fun i => And (Ne (p.coeff i) …
  -/
  refine p.recOnHorner ?_ ?_ ?_
    /-
      case refine_1
      R : Type u_1
      inst✝¹ : CommRing R
      S : Type u_2
      inst✝ : CommRing S
      f : RingHom R S
      I : Ideal S
      r : S
      r_non_zero_divisor : ∀ {x : S}, Eq (HMul.hMul x r) 0 → Eq x 0
      hr : Membership.mem I r
      p : Polynomial R
      ⊢ Ne 0 0 → Eq (Polynomial.eval₂ f r 0) 0 → Exists fun i => And (Ne (Polynomial …
    -/
  · intro h
    /-
      case refine_1
      R : Type u_1
      inst✝¹ : CommRing R
      S : Type u_2
      inst✝ : CommRing S
      f : RingHom R S
      I : Ideal S
      r : S
      r_non_zero_divisor : ∀ {x : S}, Eq (HMul.hMul x r) 0 → Eq x 0
      hr : Membership.mem I r
      p : Polynomial R
      h : Ne 0 0
      ⊢ Eq (Polynomial.eval₂ f r 0) 0 → Exists fun i => And (Ne (Polynomial.coeff 0  …
    -/
    contradiction
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      inst✝¹ : CommRing R
      S : Type u_2
      inst✝ : CommRing S
      f : RingHom R S
      I : Ideal S
      r : S
      r_non_zero_divisor : ∀ {x : S}, Eq (HMul.hMul x r) 0 → Eq x 0
      hr : Membership.mem I r
      p : Polynomial R
      ⊢ ∀ (p : Polynomial R) (a : R), Eq (p.coeff 0) 0 → Ne a 0 → (Ne p 0 → Eq (Poly …
    -/
  · intro p a coeff_eq_zero a_ne_zero _ _ hp
    /-
      case refine_2
      R : Type u_1
      inst✝¹ : CommRing R
      S : Type u_2
      inst✝ : CommRing S
      f : RingHom R S
      I : Ideal S
      r : S
      r_non_zero_divisor : ∀ {x : S}, Eq (HMul.hMul x r) 0 → Eq x 0
      hr : Membership.mem I r
      p✝ p : Polynomial R
      a : R
      coeff_eq_zero : Eq (p.coeff 0) 0
      a_ne_zero : Ne a 0
      a✝¹ : Ne p 0 → Eq (Polynomial.eval₂ f r p) 0 → Exists fun i => And (Ne (p.coef …
      a✝ : Ne (HAdd.hAdd p (Polynomial.C a)) 0
      hp : Eq (Polynomial.eval₂ f r (HAdd.hAdd p (Polynomial.C a))) 0
      ⊢ Exists fun i => And (Ne ((HAdd.hAdd p (Polynomial.C a)).coeff i) 0) (Members …
    -/
    refine ⟨0, ?_, coeff_zero_mem_comap_of_root_mem hr hp⟩
    /-
      case refine_2
      R : Type u_1
      inst✝¹ : CommRing R
      S : Type u_2
      inst✝ : CommRing S
      f : RingHom R S
      I : Ideal S
      r : S
      r_non_zero_divisor : ∀ {x : S}, Eq (HMul.hMul x r) 0 → Eq x 0
      hr : Membership.mem I r
      p✝ p : Polynomial R
      a : R
      coeff_eq_zero : Eq (p.coeff 0) 0
      a_ne_zero : Ne a 0
      a✝¹ : Ne p 0 → Eq (Polynomial.eval₂ f r p) 0 → Exists fun i => And (Ne (p.coef …
      a✝ : Ne (HAdd.hAdd p (Polynomial.C a)) 0
      hp : Eq (Polynomial.eval₂ f r (HAdd.hAdd p (Polynomial.C a))) 0
      ⊢ Ne ((HAdd.hAdd p (Polynomial.C a)).coeff 0) 0
    -/
    simp [coeff_eq_zero, a_ne_zero]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      R : Type u_1
      inst✝¹ : CommRing R
      S : Type u_2
      inst✝ : CommRing S
      f : RingHom R S
      I : Ideal S
      r : S
      r_non_zero_divisor : ∀ {x : S}, Eq (HMul.hMul x r) 0 → Eq x 0
      hr : Membership.mem I r
      p : Polynomial R
      ⊢ ∀ (p : Polynomial R), Ne p 0 → (Ne p 0 → Eq (Polynomial.eval₂ f r p) 0 → Exi …
    -/
  · intro p p_nonzero ih _ hp
    /-
      case refine_3
      R : Type u_1
      inst✝¹ : CommRing R
      S : Type u_2
      inst✝ : CommRing S
      f : RingHom R S
      I : Ideal S
      r : S
      r_non_zero_divisor : ∀ {x : S}, Eq (HMul.hMul x r) 0 → Eq x 0
      hr : Membership.mem I r
      p✝ p : Polynomial R
      p_nonzero : Ne p 0
      ih : Ne p 0 → Eq (Polynomial.eval₂ f r p) 0 → Exists fun i => And (Ne (p.coeff …
      a✝ : Ne (HMul.hMul p Polynomial.X) 0
      hp : Eq (Polynomial.eval₂ f r (HMul.hMul p Polynomial.X)) 0
      ⊢ Exists fun i => And (Ne ((HMul.hMul p Polynomial.X).coeff i) 0) (Membership. …
    -/
    rw [eval₂_mul, eval₂_X] at hp
    /-
      case refine_3
      R : Type u_1
      inst✝¹ : CommRing R
      S : Type u_2
      inst✝ : CommRing S
      f : RingHom R S
      I : Ideal S
      r : S
      r_non_zero_divisor : ∀ {x : S}, Eq (HMul.hMul x r) 0 → Eq x 0
      hr : Membership.mem I r
      p✝ p : Polynomial R
      p_nonzero : Ne p 0
      ih : Ne p 0 → Eq (Polynomial.eval₂ f r p) 0 → Exists fun i => And (Ne (p.coeff …
      a✝ : Ne (HMul.hMul p Polynomial.X) 0
      hp : Eq (HMul.hMul (Polynomial.eval₂ f r p) r) 0
      ⊢ Exists fun i => And (Ne ((HMul.hMul p Polynomial.X).coeff i) 0) (Membership. …
    -/
    obtain ⟨i, hi, mem⟩ := ih p_nonzero (r_non_zero_divisor hp)
    /-
      case refine_3.intro.intro
      R : Type u_1
      inst✝¹ : CommRing R
      S : Type u_2
      inst✝ : CommRing S
      f : RingHom R S
      I : Ideal S
      r : S
      r_non_zero_divisor : ∀ {x : S}, Eq (HMul.hMul x r) 0 → Eq x 0
      hr : Membership.mem I r
      p✝ p : Polynomial R
      p_nonzero : Ne p 0
      ih : Ne p 0 → Eq (Polynomial.eval₂ f r p) 0 → Exists fun i => And (Ne (p.coeff …
      a✝ : Ne (HMul.hMul p Polynomial.X) 0
      hp : Eq (HMul.hMul (Polynomial.eval₂ f r p) r) 0
      i : Nat
      hi : Ne (p.coeff i) 0
      mem : Membership.mem (Ideal.comap f I) (p.coeff i)
      ⊢ Exists fun i => And (Ne ((HMul.hMul p Polynomial.X).coeff i) 0) (Membership. …
    -/
    refine ⟨i + 1, ?_, ?_⟩
      /-
        case refine_3.intro.intro.refine_1
        R : Type u_1
        inst✝¹ : CommRing R
        S : Type u_2
        inst✝ : CommRing S
        f : RingHom R S
        I : Ideal S
        r : S
        r_non_zero_divisor : ∀ {x : S}, Eq (HMul.hMul x r) 0 → Eq x 0
        hr : Membership.mem I r
        p✝ p : Polynomial R
        p_nonzero : Ne p 0
        ih : Ne p 0 → Eq (Polynomial.eval₂ f r p) 0 → Exists fun i => And (Ne (p.coeff …
        a✝ : Ne (HMul.hMul p Polynomial.X) 0
        hp : Eq (HMul.hMul (Polynomial.eval₂ f r p) r) 0
        i : Nat
        hi : Ne (p.coeff i) 0
        mem : Membership.mem (Ideal.comap f I) (p.coeff i)
        ⊢ Ne ((HMul.hMul p Polynomial.X).coeff (HAdd.hAdd i 1)) 0
      -/
    · simp [hi, mem]
      /-
        🎉 no goals
      -/
      /-
        case refine_3.intro.intro.refine_2
        R : Type u_1
        inst✝¹ : CommRing R
        S : Type u_2
        inst✝ : CommRing S
        f : RingHom R S
        I : Ideal S
        r : S
        r_non_zero_divisor : ∀ {x : S}, Eq (HMul.hMul x r) 0 → Eq x 0
        hr : Membership.mem I r
        p✝ p : Polynomial R
        p_nonzero : Ne p 0
        ih : Ne p 0 → Eq (Polynomial.eval₂ f r p) 0 → Exists fun i => And (Ne (p.coeff …
        a✝ : Ne (HMul.hMul p Polynomial.X) 0
        hp : Eq (HMul.hMul (Polynomial.eval₂ f r p) r) 0
        i : Nat
        hi : Ne (p.coeff i) 0
        mem : Membership.mem (Ideal.comap f I) (p.coeff i)
        ⊢ Membership.mem (Ideal.comap f I) ((HMul.hMul p Polynomial.X).coeff (HAdd.hAd …
      -/
    · simpa [hi] using mem
      /-
        🎉 no goals
      -/


/-- Let `P` be an ideal in `R[x]`.  The map
`R[x]/P → (R / (P ∩ R))[x] / (P / (P ∩ R))`
is injective.
-/
theorem injective_quotient_le_comap_map (P : Ideal R[X]) :
    Function.Injective <|
      Ideal.quotientMap
        (Ideal.map (Polynomial.mapRingHom (Quotient.mk (P.comap (C : R →+* R[X])))) P)
        (Polynomial.mapRingHom (Ideal.Quotient.mk (P.comap (C : R →+* R[X]))))
        le_comap_map := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    P : Ideal (Polynomial R)
    ⊢ Function.Injective ⇑(Ideal.quotientMap (Ideal.map (Polynomial.mapRingHom (Id …
  -/
  refine quotientMap_injective' (le_of_eq ?_)
  rw [comap_map_of_surjective (mapRingHom (Ideal.Quotient.mk (P.comap (C : R →+* R[X]))))
      (map_surjective (Ideal.Quotient.mk (P.comap (C : R →+* R[X]))) Ideal.Quotient.mk_surjective)]
  /-
    R : Type u_1
    inst✝ : CommRing R
    P : Ideal (Polynomial R)
    ⊢ Eq (Max.max P (Ideal.comap (Polynomial.mapRingHom (Ideal.Quotient.mk (Ideal. …
  -/
  refine le_antisymm (sup_le le_rfl ?_) (le_sup_of_le_left le_rfl)
  refine fun p hp =>
    polynomial_mem_ideal_of_coeff_mem_ideal P p fun n => Ideal.Quotient.eq_zero_iff_mem.mp ?_
  /-
    R : Type u_1
    inst✝ : CommRing R
    P : Ideal (Polynomial R)
    p : Polynomial R
    hp : Membership.mem (Ideal.comap (Polynomial.mapRingHom (Ideal.Quotient.mk (Id …
    n : Nat
    ⊢ Eq ((Ideal.Quotient.mk (Ideal.comap Polynomial.C P)) (p.coeff n)) 0
  -/
  simpa only [coeff_map, coe_mapRingHom] using ext_iff.mp (Ideal.mem_bot.mp (mem_comap.mp hp)) n
  /-
    🎉 no goals
  -/


/-- The identity in this lemma asserts that the "obvious" square
```
    R    → (R / (P ∩ R))
    ↓          ↓
R[x] / P → (R / (P ∩ R))[x] / (P / (P ∩ R))
```
commutes.  It is used, for instance, in the proof of `quotient_mk_comp_C_is_integral_of_jacobson`,
in the file `Mathlib.RingTheory.Jacobson.Polynomial`.
-/
theorem quotient_mk_maps_eq (P : Ideal R[X]) :
    ((Quotient.mk (map (mapRingHom (Quotient.mk (P.comap (C : R →+* R[X])))) P)).comp C).comp
        (Quotient.mk (P.comap (C : R →+* R[X]))) =
      (Ideal.quotientMap (map (mapRingHom (Quotient.mk (P.comap (C : R →+* R[X])))) P)
            (mapRingHom (Quotient.mk (P.comap (C : R →+* R[X])))) le_comap_map).comp
        ((Quotient.mk P).comp C) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    P : Ideal (Polynomial R)
    ⊢ Eq (((Ideal.Quotient.mk (Ideal.map (Polynomial.mapRingHom (Ideal.Quotient.mk …
  -/
  refine RingHom.ext fun x => ?_
  /-
    R : Type u_1
    inst✝ : CommRing R
    P : Ideal (Polynomial R)
    x : R
    ⊢ Eq ((((Ideal.Quotient.mk (Ideal.map (Polynomial.mapRingHom (Ideal.Quotient.m …
  -/
  repeat' rw [RingHom.coe_comp, Function.comp_apply]
  /-
    R : Type u_1
    inst✝ : CommRing R
    P : Ideal (Polynomial R)
    x : R
    ⊢ Eq ((Ideal.Quotient.mk (Ideal.map (Polynomial.mapRingHom (Ideal.Quotient.mk  …
  -/
  rw [quotientMap_mk, coe_mapRingHom, map_C]
  /-
    🎉 no goals
  -/


/-- This technical lemma asserts the existence of a polynomial `p` in an ideal `P ⊂ R[x]`
that is non-zero in the quotient `R / (P ∩ R) [x]`.  The assumptions are equivalent to
`P ≠ 0` and `P ∩ R = (0)`.
-/
theorem exists_nonzero_mem_of_ne_bot {P : Ideal R[X]} (Pb : P ≠ ⊥) (hP : ∀ x : R, C x ∈ P → x = 0) :
    ∃ p : R[X], p ∈ P ∧ Polynomial.map (Quotient.mk (P.comap (C : R →+* R[X]))) p ≠ 0 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    P : Ideal (Polynomial R)
    Pb : Ne P Bot.bot
    hP : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
    ⊢ Exists fun p => And (Membership.mem P p) (Ne (Polynomial.map (Ideal.Quotient …
  -/
  obtain ⟨m, hm⟩ := Submodule.nonzero_mem_of_bot_lt (bot_lt_iff_ne_bot.mpr Pb)
  /-
    case intro
    R : Type u_1
    inst✝ : CommRing R
    P : Ideal (Polynomial R)
    Pb : Ne P Bot.bot
    hP : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
    m : Subtype fun x => Membership.mem P x
    hm : Ne m 0
    ⊢ Exists fun p => And (Membership.mem P p) (Ne (Polynomial.map (Ideal.Quotient …
  -/
  refine ⟨m, Submodule.coe_mem m, fun pp0 => hm (Submodule.coe_eq_zero.mp ?_)⟩
  refine
    (injective_iff_map_eq_zero (Polynomial.mapRingHom (Ideal.Quotient.mk
      (P.comap (C : R →+* R[X]))))).mp
      ?_ _ pp0
  /-
    case intro
    R : Type u_1
    inst✝ : CommRing R
    P : Ideal (Polynomial R)
    Pb : Ne P Bot.bot
    hP : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
    m : Subtype fun x => Membership.mem P x
    hm : Ne m 0
    pp0 : Eq (Polynomial.map (Ideal.Quotient.mk (Ideal.comap Polynomial.C P)) ↑m) 0
    ⊢ Function.Injective ⇑(Polynomial.mapRingHom (Ideal.Quotient.mk (Ideal.comap P …
  -/
  refine map_injective _ ((Ideal.Quotient.mk (P.comap C)).injective_iff_ker_eq_bot.mpr ?_)
  /-
    case intro
    R : Type u_1
    inst✝ : CommRing R
    P : Ideal (Polynomial R)
    Pb : Ne P Bot.bot
    hP : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
    m : Subtype fun x => Membership.mem P x
    hm : Ne m 0
    pp0 : Eq (Polynomial.map (Ideal.Quotient.mk (Ideal.comap Polynomial.C P)) ↑m) 0
    ⊢ Eq (RingHom.ker (Ideal.Quotient.mk (Ideal.comap Polynomial.C P))) Bot.bot
  -/
  rw [mk_ker]
  /-
    case intro
    R : Type u_1
    inst✝ : CommRing R
    P : Ideal (Polynomial R)
    Pb : Ne P Bot.bot
    hP : ∀ (x : R), Membership.mem P (Polynomial.C x) → Eq x 0
    m : Subtype fun x => Membership.mem P x
    hm : Ne m 0
    pp0 : Eq (Polynomial.map (Ideal.Quotient.mk (Ideal.comap Polynomial.C P)) ↑m) 0
    ⊢ Eq (Ideal.comap Polynomial.C P) Bot.bot
  -/
  exact (Submodule.eq_bot_iff _).mpr fun x hx => hP x (mem_comap.mp hx)
  /-
    🎉 no goals
  -/


/-- If there is an injective map `R/p → S/P` such that following diagram commutes:
```
R   → S
↓     ↓
R/p → S/P
```
then `P` lies over `p`.
-/
theorem comap_eq_of_scalar_tower_quotient [Algebra R S] [Algebra (R ⧸ p) (S ⧸ P)]
    [IsScalarTower R (R ⧸ p) (S ⧸ P)] (h : Function.Injective (algebraMap (R ⧸ p) (S ⧸ P))) :
    comap (algebraMap R S) P = p := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    S : Type u_2
    inst✝³ : CommRing S
    p : Ideal R
    P : Ideal S
    inst✝² : Algebra R S
    inst✝¹ : Algebra (HasQuotient.Quotient R p) (HasQuotient.Quotient S P)
    inst✝ : IsScalarTower R (HasQuotient.Quotient R p) (HasQuotient.Quotient S P)
    h : Function.Injective ⇑(algebraMap (HasQuotient.Quotient R p) (HasQuotient.Qu …
    ⊢ Eq (Ideal.comap (algebraMap R S) P) p
  -/
  ext x
  rw [mem_comap, ← Quotient.eq_zero_iff_mem, ← Quotient.eq_zero_iff_mem, Quotient.mk_algebraMap,
    IsScalarTower.algebraMap_apply R (R ⧸ p) (S ⧸ P), Quotient.algebraMap_eq]
  /-
    case h
    R : Type u_1
    inst✝⁴ : CommRing R
    S : Type u_2
    inst✝³ : CommRing S
    p : Ideal R
    P : Ideal S
    inst✝² : Algebra R S
    inst✝¹ : Algebra (HasQuotient.Quotient R p) (HasQuotient.Quotient S P)
    inst✝ : IsScalarTower R (HasQuotient.Quotient R p) (HasQuotient.Quotient S P)
    h : Function.Injective ⇑(algebraMap (HasQuotient.Quotient R p) (HasQuotient.Qu …
    x : R
    ⊢ Iff (Eq ((algebraMap (HasQuotient.Quotient R p) (HasQuotient.Quotient S P))  …
  -/
  constructor
    /-
      case h.mp
      R : Type u_1
      inst✝⁴ : CommRing R
      S : Type u_2
      inst✝³ : CommRing S
      p : Ideal R
      P : Ideal S
      inst✝² : Algebra R S
      inst✝¹ : Algebra (HasQuotient.Quotient R p) (HasQuotient.Quotient S P)
      inst✝ : IsScalarTower R (HasQuotient.Quotient R p) (HasQuotient.Quotient S P)
      h : Function.Injective ⇑(algebraMap (HasQuotient.Quotient R p) (HasQuotient.Qu …
      x : R
      ⊢ Eq ((algebraMap (HasQuotient.Quotient R p) (HasQuotient.Quotient S P)) ((Ide …
    -/
  · intro hx
    /-
      case h.mp
      R : Type u_1
      inst✝⁴ : CommRing R
      S : Type u_2
      inst✝³ : CommRing S
      p : Ideal R
      P : Ideal S
      inst✝² : Algebra R S
      inst✝¹ : Algebra (HasQuotient.Quotient R p) (HasQuotient.Quotient S P)
      inst✝ : IsScalarTower R (HasQuotient.Quotient R p) (HasQuotient.Quotient S P)
      h : Function.Injective ⇑(algebraMap (HasQuotient.Quotient R p) (HasQuotient.Qu …
      x : R
      hx : Eq ((algebraMap (HasQuotient.Quotient R p) (HasQuotient.Quotient S P)) (( …
      ⊢ Eq ((Ideal.Quotient.mk p) x) 0
    -/
    exact (injective_iff_map_eq_zero (algebraMap (R ⧸ p) (S ⧸ P))).mp h _ hx
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      R : Type u_1
      inst✝⁴ : CommRing R
      S : Type u_2
      inst✝³ : CommRing S
      p : Ideal R
      P : Ideal S
      inst✝² : Algebra R S
      inst✝¹ : Algebra (HasQuotient.Quotient R p) (HasQuotient.Quotient S P)
      inst✝ : IsScalarTower R (HasQuotient.Quotient R p) (HasQuotient.Quotient S P)
      h : Function.Injective ⇑(algebraMap (HasQuotient.Quotient R p) (HasQuotient.Qu …
      x : R
      ⊢ Eq ((Ideal.Quotient.mk p) x) 0 → Eq ((algebraMap (HasQuotient.Quotient R p)  …
    -/
  · intro hx
    /-
      case h.mpr
      R : Type u_1
      inst✝⁴ : CommRing R
      S : Type u_2
      inst✝³ : CommRing S
      p : Ideal R
      P : Ideal S
      inst✝² : Algebra R S
      inst✝¹ : Algebra (HasQuotient.Quotient R p) (HasQuotient.Quotient S P)
      inst✝ : IsScalarTower R (HasQuotient.Quotient R p) (HasQuotient.Quotient S P)
      h : Function.Injective ⇑(algebraMap (HasQuotient.Quotient R p) (HasQuotient.Qu …
      x : R
      hx : Eq ((Ideal.Quotient.mk p) x) 0
      ⊢ Eq ((algebraMap (HasQuotient.Quotient R p) (HasQuotient.Quotient S P)) ((Ide …
    -/
    rw [hx, RingHom.map_zero]
    /-
      🎉 no goals
    -/


/-- `R / p` has a canonical map to `S / pS`. -/
instance Quotient.algebraQuotientMapQuotient : Algebra (R ⧸ p) (S ⧸ map (algebraMap R S) p) :=
  Ideal.Quotient.algebraQuotientOfLEComap le_comap_map


@[simp]
theorem Quotient.algebraMap_quotient_map_quotient (x : R) :
    letI f := algebraMap R S
    algebraMap (R ⧸ p) (S ⧸ map f p) (Ideal.Quotient.mk p x) =
    Ideal.Quotient.mk (map f p) (f x) :=
  rfl


@[simp]
theorem Quotient.mk_smul_mk_quotient_map_quotient (x : R) (y : S) :
    letI f := algebraMap R S
    Quotient.mk p x • Quotient.mk (map f p) y = Quotient.mk (map f p) (f x * y) :=
  Algebra.smul_def _ _


instance Quotient.tower_quotient_map_quotient [Algebra R S] :
    IsScalarTower R (R ⧸ p) (S ⧸ map (algebraMap R S) p) :=
  IsScalarTower.of_algebraMap_eq fun x => by
    rw [Quotient.algebraMap_eq, Quotient.algebraMap_quotient_map_quotient,
      Quotient.mk_algebraMap]


instance QuotientMapQuotient.isNoetherian [Algebra R S] [IsNoetherian R S] (I : Ideal R) :
    IsNoetherian (R ⧸ I) (S ⧸ I.map (algebraMap R S)) :=
  isNoetherian_of_tower R <|
    isNoetherian_of_surjective S (Ideal.Quotient.mkₐ R _).toLinearMap <|
      LinearMap.range_eq_top.mpr Ideal.Quotient.mk_surjective


theorem exists_coeff_ne_zero_mem_comap_of_root_mem [IsDomain S] {r : S} (r_ne_zero : r ≠ 0)
    (hr : r ∈ I) {p : R[X]} :
    p ≠ 0 → p.eval₂ f r = 0 → ∃ i, p.coeff i ≠ 0 ∧ p.coeff i ∈ I.comap f :=
  exists_coeff_ne_zero_mem_comap_of_non_zero_divisor_root_mem
    (fun {_} h => Or.resolve_right (mul_eq_zero.mp h) r_ne_zero) hr


theorem exists_coeff_mem_comap_sdiff_comap_of_root_mem_sdiff [IsPrime I] (hIJ : I ≤ J) {r : S}
    (hr : r ∈ (J : Set S) \ I) {p : R[X]} (p_ne_zero : p.map (Quotient.mk (I.comap f)) ≠ 0)
    (hpI : p.eval₂ f r ∈ I) : ∃ i, p.coeff i ∈ (J.comap f : Set R) \ I.comap f := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Type u_2
    inst✝¹ : CommRing S
    f : RingHom R S
    I J : Ideal S
    inst✝ : I.IsPrime
    hIJ : LE.le I J
    r : S
    hr : Membership.mem (SDiff.sdiff ↑J ↑I) r
    p : Polynomial R
    p_ne_zero : Ne (Polynomial.map (Ideal.Quotient.mk (Ideal.comap f I)) p) 0
    hpI : Membership.mem I (Polynomial.eval₂ f r p)
    ⊢ Exists fun i => Membership.mem (SDiff.sdiff ↑(Ideal.comap f J) ↑(Ideal.comap …
  -/
  obtain ⟨hrJ, hrI⟩ := hr
  /-
    case intro
    R : Type u_1
    inst✝² : CommRing R
    S : Type u_2
    inst✝¹ : CommRing S
    f : RingHom R S
    I J : Ideal S
    inst✝ : I.IsPrime
    hIJ : LE.le I J
    r : S
    p : Polynomial R
    p_ne_zero : Ne (Polynomial.map (Ideal.Quotient.mk (Ideal.comap f I)) p) 0
    hpI : Membership.mem I (Polynomial.eval₂ f r p)
    hrJ : Membership.mem (↑J) r
    hrI : Not (Membership.mem (↑I) r)
    ⊢ Exists fun i => Membership.mem (SDiff.sdiff ↑(Ideal.comap f J) ↑(Ideal.comap …
  -/
  have rbar_ne_zero : Ideal.Quotient.mk I r ≠ 0 := mt (Quotient.mk_eq_zero I).mp hrI
  /-
    case intro
    R : Type u_1
    inst✝² : CommRing R
    S : Type u_2
    inst✝¹ : CommRing S
    f : RingHom R S
    I J : Ideal S
    inst✝ : I.IsPrime
    hIJ : LE.le I J
    r : S
    p : Polynomial R
    p_ne_zero : Ne (Polynomial.map (Ideal.Quotient.mk (Ideal.comap f I)) p) 0
    hpI : Membership.mem I (Polynomial.eval₂ f r p)
    hrJ : Membership.mem (↑J) r
    hrI : Not (Membership.mem (↑I) r)
    rbar_ne_zero : Ne ((Ideal.Quotient.mk I) r) 0
    ⊢ Exists fun i => Membership.mem (SDiff.sdiff ↑(Ideal.comap f J) ↑(Ideal.comap …
  -/
  have rbar_mem_J : Ideal.Quotient.mk I r ∈ J.map (Ideal.Quotient.mk I) := mem_map_of_mem _ hrJ
  have quotient_f : ∀ x ∈ I.comap f, (Ideal.Quotient.mk I).comp f x = 0 := by
    simp [Quotient.eq_zero_iff_mem]
  have rbar_root :
    (p.map (Ideal.Quotient.mk (I.comap f))).eval₂ (Quotient.lift (I.comap f) _ quotient_f)
        (Ideal.Quotient.mk I r) =
      0 := by
    convert Quotient.eq_zero_iff_mem.mpr hpI
    exact _root_.trans (eval₂_map _ _ _) (hom_eval₂ p f (Ideal.Quotient.mk I) r).symm
  obtain ⟨i, ne_zero, mem⟩ :=
    exists_coeff_ne_zero_mem_comap_of_root_mem rbar_ne_zero rbar_mem_J p_ne_zero rbar_root
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝² : CommRing R
    S : Type u_2
    inst✝¹ : CommRing S
    f : RingHom R S
    I J : Ideal S
    inst✝ : I.IsPrime
    hIJ : LE.le I J
    r : S
    p : Polynomial R
    p_ne_zero : Ne (Polynomial.map (Ideal.Quotient.mk (Ideal.comap f I)) p) 0
    hpI : Membership.mem I (Polynomial.eval₂ f r p)
    hrJ : Membership.mem (↑J) r
    hrI : Not (Membership.mem (↑I) r)
    rbar_ne_zero : Ne ((Ideal.Quotient.mk I) r) 0
    rbar_mem_J : Membership.mem (Ideal.map (Ideal.Quotient.mk I) J) ((Ideal.Quotie …
    quotient_f : ∀ (x : R), Membership.mem (Ideal.comap f I) x → Eq (((Ideal.Quoti …
    rbar_root : Eq (Polynomial.eval₂ (Ideal.Quotient.lift (Ideal.comap f I) ((Idea …
    i : Nat
    ne_zero : Ne ((Polynomial.map (Ideal.Quotient.mk (Ideal.comap f I)) p).coeff i …
    mem : Membership.mem (Ideal.comap (Ideal.Quotient.lift (Ideal.comap f I) ((Ide …
    ⊢ Exists fun i => Membership.mem (SDiff.sdiff ↑(Ideal.comap f J) ↑(Ideal.comap …
  -/
  rw [coeff_map] at ne_zero mem
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝² : CommRing R
    S : Type u_2
    inst✝¹ : CommRing S
    f : RingHom R S
    I J : Ideal S
    inst✝ : I.IsPrime
    hIJ : LE.le I J
    r : S
    p : Polynomial R
    p_ne_zero : Ne (Polynomial.map (Ideal.Quotient.mk (Ideal.comap f I)) p) 0
    hpI : Membership.mem I (Polynomial.eval₂ f r p)
    hrJ : Membership.mem (↑J) r
    hrI : Not (Membership.mem (↑I) r)
    rbar_ne_zero : Ne ((Ideal.Quotient.mk I) r) 0
    rbar_mem_J : Membership.mem (Ideal.map (Ideal.Quotient.mk I) J) ((Ideal.Quotie …
    quotient_f : ∀ (x : R), Membership.mem (Ideal.comap f I) x → Eq (((Ideal.Quoti …
    rbar_root : Eq (Polynomial.eval₂ (Ideal.Quotient.lift (Ideal.comap f I) ((Idea …
    i : Nat
    ne_zero : Ne ((Ideal.Quotient.mk (Ideal.comap f I)) (p.coeff i)) 0
    mem : Membership.mem (Ideal.comap (Ideal.Quotient.lift (Ideal.comap f I) ((Ide …
    ⊢ Exists fun i => Membership.mem (SDiff.sdiff ↑(Ideal.comap f J) ↑(Ideal.comap …
  -/
  refine ⟨i, (mem_quotient_iff_mem hIJ).mp ?_, mt ?_ ne_zero⟩
    /-
      case intro.intro.intro.refine_1
      R : Type u_1
      inst✝² : CommRing R
      S : Type u_2
      inst✝¹ : CommRing S
      f : RingHom R S
      I J : Ideal S
      inst✝ : I.IsPrime
      hIJ : LE.le I J
      r : S
      p : Polynomial R
      p_ne_zero : Ne (Polynomial.map (Ideal.Quotient.mk (Ideal.comap f I)) p) 0
      hpI : Membership.mem I (Polynomial.eval₂ f r p)
      hrJ : Membership.mem (↑J) r
      hrI : Not (Membership.mem (↑I) r)
      rbar_ne_zero : Ne ((Ideal.Quotient.mk I) r) 0
      rbar_mem_J : Membership.mem (Ideal.map (Ideal.Quotient.mk I) J) ((Ideal.Quotie …
      quotient_f : ∀ (x : R), Membership.mem (Ideal.comap f I) x → Eq (((Ideal.Quoti …
      rbar_root : Eq (Polynomial.eval₂ (Ideal.Quotient.lift (Ideal.comap f I) ((Idea …
      i : Nat
      ne_zero : Ne ((Ideal.Quotient.mk (Ideal.comap f I)) (p.coeff i)) 0
      mem : Membership.mem (Ideal.comap (Ideal.Quotient.lift (Ideal.comap f I) ((Ide …
      ⊢ Membership.mem (Ideal.map (Ideal.Quotient.mk I) J) ((Ideal.Quotient.mk I) (f …
    -/
  · simpa using mem
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.intro.refine_2
    R : Type u_1
    inst✝² : CommRing R
    S : Type u_2
    inst✝¹ : CommRing S
    f : RingHom R S
    I J : Ideal S
    inst✝ : I.IsPrime
    hIJ : LE.le I J
    r : S
    p : Polynomial R
    p_ne_zero : Ne (Polynomial.map (Ideal.Quotient.mk (Ideal.comap f I)) p) 0
    hpI : Membership.mem I (Polynomial.eval₂ f r p)
    hrJ : Membership.mem (↑J) r
    hrI : Not (Membership.mem (↑I) r)
    rbar_ne_zero : Ne ((Ideal.Quotient.mk I) r) 0
    rbar_mem_J : Membership.mem (Ideal.map (Ideal.Quotient.mk I) J) ((Ideal.Quotie …
    quotient_f : ∀ (x : R), Membership.mem (Ideal.comap f I) x → Eq (((Ideal.Quoti …
    rbar_root : Eq (Polynomial.eval₂ (Ideal.Quotient.lift (Ideal.comap f I) ((Idea …
    i : Nat
    ne_zero : Ne ((Ideal.Quotient.mk (Ideal.comap f I)) (p.coeff i)) 0
    mem : Membership.mem (Ideal.comap (Ideal.Quotient.lift (Ideal.comap f I) ((Ide …
    ⊢ Membership.mem (↑(Ideal.comap f I)) (p.coeff i) → Eq ((Ideal.Quotient.mk (Id …
  -/
  simp [Quotient.eq_zero_iff_mem]
  /-
    🎉 no goals
  -/


theorem comap_lt_comap_of_root_mem_sdiff [I.IsPrime] (hIJ : I ≤ J) {r : S}
    (hr : r ∈ (J : Set S) \ I) {p : R[X]} (p_ne_zero : p.map (Quotient.mk (I.comap f)) ≠ 0)
    (hp : p.eval₂ f r ∈ I) : I.comap f < J.comap f :=
  let ⟨i, hJ, hI⟩ := exists_coeff_mem_comap_sdiff_comap_of_root_mem_sdiff hIJ hr p_ne_zero hp
  SetLike.lt_iff_le_and_exists.mpr ⟨comap_mono hIJ, p.coeff i, hJ, hI⟩


theorem mem_of_one_mem (h : (1 : S) ∈ I) (x) : x ∈ I :=
  (I.eq_top_iff_one.mpr h).symm ▸ mem_top


theorem comap_lt_comap_of_integral_mem_sdiff [Algebra R S] [hI : I.IsPrime] (hIJ : I ≤ J) {x : S}
    (mem : x ∈ (J : Set S) \ I) (integral : IsIntegral R x) :
    I.comap (algebraMap R S) < J.comap (algebraMap R S) := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Type u_2
    inst✝¹ : CommRing S
    I J : Ideal S
    inst✝ : Algebra R S
    hI : I.IsPrime
    hIJ : LE.le I J
    x : S
    mem : Membership.mem (SDiff.sdiff ↑J ↑I) x
    integral : IsIntegral R x
    ⊢ LT.lt (Ideal.comap (algebraMap R S) I) (Ideal.comap (algebraMap R S) J)
  -/
  obtain ⟨p, p_monic, hpx⟩ := integral
  /-
    case intro.intro
    R : Type u_1
    inst✝² : CommRing R
    S : Type u_2
    inst✝¹ : CommRing S
    I J : Ideal S
    inst✝ : Algebra R S
    hI : I.IsPrime
    hIJ : LE.le I J
    x : S
    mem : Membership.mem (SDiff.sdiff ↑J ↑I) x
    p : Polynomial R
    p_monic : p.Monic
    hpx : Eq (Polynomial.eval₂ (algebraMap R S) x p) 0
    ⊢ LT.lt (Ideal.comap (algebraMap R S) I) (Ideal.comap (algebraMap R S) J)
  -/
  refine comap_lt_comap_of_root_mem_sdiff hIJ mem (map_monic_ne_zero p_monic) ?_
  /-
    case intro.intro
    R : Type u_1
    inst✝² : CommRing R
    S : Type u_2
    inst✝¹ : CommRing S
    I J : Ideal S
    inst✝ : Algebra R S
    hI : I.IsPrime
    hIJ : LE.le I J
    x : S
    mem : Membership.mem (SDiff.sdiff ↑J ↑I) x
    p : Polynomial R
    p_monic : p.Monic
    hpx : Eq (Polynomial.eval₂ (algebraMap R S) x p) 0
    ⊢ Membership.mem I (Polynomial.eval₂ (algebraMap R S) x p)
  -/
  convert I.zero_mem
  /-
    🎉 no goals
  -/


theorem comap_ne_bot_of_root_mem [IsDomain S] {r : S} (r_ne_zero : r ≠ 0) (hr : r ∈ I) {p : R[X]}
    (p_ne_zero : p ≠ 0) (hp : p.eval₂ f r = 0) : I.comap f ≠ ⊥ := fun h =>
  let ⟨_, hi, mem⟩ := exists_coeff_ne_zero_mem_comap_of_root_mem r_ne_zero hr p_ne_zero hp
  absurd (mem_bot.mp (eq_bot_iff.mp h mem)) hi


theorem isMaximal_of_isIntegral_of_isMaximal_comap [Algebra R S] [Algebra.IsIntegral R S]
    (I : Ideal S) [I.IsPrime] (hI : IsMaximal (I.comap (algebraMap R S))) : IsMaximal I :=
  ⟨⟨mt comap_eq_top_iff.mpr hI.1.1, fun _ I_lt_J =>
      let ⟨I_le_J, x, hxJ, hxI⟩ := SetLike.lt_iff_le_and_exists.mp I_lt_J
      comap_eq_top_iff.1 <|
        hI.1.2 _ (comap_lt_comap_of_integral_mem_sdiff I_le_J ⟨hxJ, hxI⟩
          (Algebra.IsIntegral.isIntegral x))⟩⟩


theorem isMaximal_of_isIntegral_of_isMaximal_comap' (f : R →+* S) (hf : f.IsIntegral) (I : Ideal S)
    [I.IsPrime] (hI : IsMaximal (I.comap f)) : IsMaximal I :=
  let _ : Algebra R S := f.toAlgebra
  have : Algebra.IsIntegral R S := ⟨hf⟩
  isMaximal_of_isIntegral_of_isMaximal_comap (R := R) (S := S) I hI


theorem comap_ne_bot_of_algebraic_mem [IsDomain S] {x : S} (x_ne_zero : x ≠ 0) (x_mem : x ∈ I)
    (hx : IsAlgebraic R x) : I.comap (algebraMap R S) ≠ ⊥ :=
  let ⟨_, p_ne_zero, hp⟩ := hx
  comap_ne_bot_of_root_mem x_ne_zero x_mem p_ne_zero hp


theorem comap_ne_bot_of_integral_mem [Nontrivial R] [IsDomain S] {x : S} (x_ne_zero : x ≠ 0)
    (x_mem : x ∈ I) (hx : IsIntegral R x) : I.comap (algebraMap R S) ≠ ⊥ :=
  comap_ne_bot_of_algebraic_mem x_ne_zero x_mem hx.isAlgebraic


theorem eq_bot_of_comap_eq_bot [Nontrivial R] [IsDomain S] [Algebra.IsIntegral R S]
    (hI : I.comap (algebraMap R S) = ⊥) : I = ⊥ := by
  /-
    R : Type u_1
    inst✝⁵ : CommRing R
    S : Type u_2
    inst✝⁴ : CommRing S
    I : Ideal S
    inst✝³ : Algebra R S
    inst✝² : Nontrivial R
    inst✝¹ : IsDomain S
    inst✝ : Algebra.IsIntegral R S
    hI : Eq (Ideal.comap (algebraMap R S) I) Bot.bot
    ⊢ Eq I Bot.bot
  -/
  refine eq_bot_iff.2 fun x hx => ?_
  /-
    R : Type u_1
    inst✝⁵ : CommRing R
    S : Type u_2
    inst✝⁴ : CommRing S
    I : Ideal S
    inst✝³ : Algebra R S
    inst✝² : Nontrivial R
    inst✝¹ : IsDomain S
    inst✝ : Algebra.IsIntegral R S
    hI : Eq (Ideal.comap (algebraMap R S) I) Bot.bot
    x : S
    hx : Membership.mem I x
    ⊢ Membership.mem Bot.bot x
  -/
  by_cases hx0 : x = 0
    /-
      case pos
      R : Type u_1
      inst✝⁵ : CommRing R
      S : Type u_2
      inst✝⁴ : CommRing S
      I : Ideal S
      inst✝³ : Algebra R S
      inst✝² : Nontrivial R
      inst✝¹ : IsDomain S
      inst✝ : Algebra.IsIntegral R S
      hI : Eq (Ideal.comap (algebraMap R S) I) Bot.bot
      x : S
      hx : Membership.mem I x
      hx0 : Eq x 0
      ⊢ Membership.mem Bot.bot x
    -/
  · exact hx0.symm ▸ Ideal.zero_mem ⊥
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝⁵ : CommRing R
      S : Type u_2
      inst✝⁴ : CommRing S
      I : Ideal S
      inst✝³ : Algebra R S
      inst✝² : Nontrivial R
      inst✝¹ : IsDomain S
      inst✝ : Algebra.IsIntegral R S
      hI : Eq (Ideal.comap (algebraMap R S) I) Bot.bot
      x : S
      hx : Membership.mem I x
      hx0 : Not (Eq x 0)
      ⊢ Membership.mem Bot.bot x
    -/
  · exact absurd hI (comap_ne_bot_of_integral_mem hx0 hx (Algebra.IsIntegral.isIntegral x))
    /-
      🎉 no goals
    -/


theorem isMaximal_comap_of_isIntegral_of_isMaximal [Algebra.IsIntegral R S] (I : Ideal S)
    [hI : I.IsMaximal] : IsMaximal (I.comap (algebraMap R S)) := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : Algebra.IsIntegral R S
    I : Ideal S
    hI : I.IsMaximal
    ⊢ (Ideal.comap (algebraMap R S) I).IsMaximal
  -/
  refine Ideal.Quotient.maximal_of_isField _ ?_
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : Algebra.IsIntegral R S
    I : Ideal S
    hI : I.IsMaximal
    ⊢ IsField (HasQuotient.Quotient R (Ideal.comap (algebraMap R S) I))
  -/
  haveI : IsPrime (I.comap (algebraMap R S)) := comap_isPrime _ _
  exact isField_of_isIntegral_of_isField
    algebraMap_quotient_injective (by rwa [← Quotient.maximal_ideal_iff_isField_quotient])


theorem isMaximal_comap_of_isIntegral_of_isMaximal' {R S : Type*} [CommRing R] [CommRing S]
    (f : R →+* S) (hf : f.IsIntegral) (I : Ideal S) [I.IsMaximal] : IsMaximal (I.comap f) :=
  let _ : Algebra R S := f.toAlgebra
  have : Algebra.IsIntegral R S := ⟨hf⟩
  isMaximal_comap_of_isIntegral_of_isMaximal (R := R) (S := S) I


theorem IsIntegralClosure.comap_lt_comap {I J : Ideal A} [I.IsPrime] (I_lt_J : I < J) :
    I.comap (algebraMap R A) < J.comap (algebraMap R A) :=
  let ⟨I_le_J, x, hxJ, hxI⟩ := SetLike.lt_iff_le_and_exists.mp I_lt_J
  comap_lt_comap_of_integral_mem_sdiff I_le_J ⟨hxJ, hxI⟩ (IsIntegralClosure.isIntegral R S x)


theorem IsIntegralClosure.isMaximal_of_isMaximal_comap (I : Ideal A) [I.IsPrime]
    (hI : IsMaximal (I.comap (algebraMap R A))) : IsMaximal I :=
  have : Algebra.IsIntegral R A := IsIntegralClosure.isIntegral_algebra R S
  isMaximal_of_isIntegral_of_isMaximal_comap I hI


theorem IsIntegralClosure.comap_ne_bot [Nontrivial R] {I : Ideal A} (I_ne_bot : I ≠ ⊥) :
    I.comap (algebraMap R A) ≠ ⊥ :=
  let ⟨x, x_mem, x_ne_zero⟩ := I.ne_bot_iff.mp I_ne_bot
  comap_ne_bot_of_integral_mem x_ne_zero x_mem (IsIntegralClosure.isIntegral R S x)


theorem IsIntegralClosure.eq_bot_of_comap_eq_bot [Nontrivial R] {I : Ideal A} :
    I.comap (algebraMap R A) = ⊥ → I = ⊥ := by
  -- Porting note: `imp_of_not_imp_not` seems not existing
  /-
    R : Type u_1
    inst✝⁹ : CommRing R
    S : Type u_2
    inst✝⁸ : CommRing S
    inst✝⁷ : Algebra R S
    A : Type u_3
    inst✝⁶ : CommRing A
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra A S
    inst✝³ : IsScalarTower R A S
    inst✝² : IsIntegralClosure A R S
    inst✝¹ : IsDomain A
    inst✝ : Nontrivial R
    I : Ideal A
    ⊢ Eq (Ideal.comap (algebraMap R A) I) Bot.bot → Eq I Bot.bot
  -/
  contrapose; exact (IsIntegralClosure.comap_ne_bot S)
              /-
                🎉 no goals
              -/


theorem IntegralClosure.comap_lt_comap {I J : Ideal (integralClosure R S)} [I.IsPrime]
    (I_lt_J : I < J) :
    I.comap (algebraMap R (integralClosure R S)) < J.comap (algebraMap R (integralClosure R S)) :=
  IsIntegralClosure.comap_lt_comap S I_lt_J


theorem IntegralClosure.isMaximal_of_isMaximal_comap (I : Ideal (integralClosure R S)) [I.IsPrime]
    (hI : IsMaximal (I.comap (algebraMap R (integralClosure R S)))) : IsMaximal I :=
  IsIntegralClosure.isMaximal_of_isMaximal_comap S I hI


theorem IntegralClosure.comap_ne_bot [Nontrivial R] {I : Ideal (integralClosure R S)}
    (I_ne_bot : I ≠ ⊥) : I.comap (algebraMap R (integralClosure R S)) ≠ ⊥ :=
  IsIntegralClosure.comap_ne_bot S I_ne_bot


theorem IntegralClosure.eq_bot_of_comap_eq_bot [Nontrivial R] {I : Ideal (integralClosure R S)} :
    I.comap (algebraMap R (integralClosure R S)) = ⊥ → I = ⊥ :=
  IsIntegralClosure.eq_bot_of_comap_eq_bot S


/-- `comap (algebraMap R S)` is a surjection from the prime spec of `R` to prime spec of `S`.
`hP : (algebraMap R S).ker ≤ P` is a slight generalization of the extension being injective -/
theorem exists_ideal_over_prime_of_isIntegral_of_isDomain [Algebra.IsIntegral R S] (P : Ideal R)
    [IsPrime P] (hP : RingHom.ker (algebraMap R S) ≤ P) :
    ∃ Q : Ideal S, IsPrime Q ∧ Q.comap (algebraMap R S) = P := by
  have hP0 : (0 : S) ∉ Algebra.algebraMapSubmonoid S P.primeCompl := by
    rintro ⟨x, ⟨hx, x0⟩⟩
    exact absurd (hP x0) hx
  /-
    R : Type u_1
    inst✝⁵ : CommRing R
    S : Type u_2
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R S
    inst✝² : IsDomain S
    inst✝¹ : Algebra.IsIntegral R S
    P : Ideal R
    inst✝ : P.IsPrime
    hP : LE.le (RingHom.ker (algebraMap R S)) P
    hP0 : Not (Membership.mem (Algebra.algebraMapSubmonoid S P.primeCompl) 0)
    ⊢ Exists fun Q => And Q.IsPrime (Eq (Ideal.comap (algebraMap R S) Q) P)
  -/
  let Rₚ := Localization P.primeCompl
  /-
    R : Type u_1
    inst✝⁵ : CommRing R
    S : Type u_2
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R S
    inst✝² : IsDomain S
    inst✝¹ : Algebra.IsIntegral R S
    P : Ideal R
    inst✝ : P.IsPrime
    hP : LE.le (RingHom.ker (algebraMap R S)) P
    hP0 : Not (Membership.mem (Algebra.algebraMapSubmonoid S P.primeCompl) 0)
    Rₚ : Type u_1 := Localization P.primeCompl
    ⊢ Exists fun Q => And Q.IsPrime (Eq (Ideal.comap (algebraMap R S) Q) P)
  -/
  let Sₚ := Localization (Algebra.algebraMapSubmonoid S P.primeCompl)
  letI : IsDomain (Localization (Algebra.algebraMapSubmonoid S P.primeCompl)) :=
    IsLocalization.isDomain_localization (le_nonZeroDivisors_of_noZeroDivisors hP0)
  /-
    R : Type u_1
    inst✝⁵ : CommRing R
    S : Type u_2
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R S
    inst✝² : IsDomain S
    inst✝¹ : Algebra.IsIntegral R S
    P : Ideal R
    inst✝ : P.IsPrime
    hP : LE.le (RingHom.ker (algebraMap R S)) P
    hP0 : Not (Membership.mem (Algebra.algebraMapSubmonoid S P.primeCompl) 0)
    Rₚ : Type u_1 := Localization P.primeCompl
    Sₚ : Type u_2 := Localization (Algebra.algebraMapSubmonoid S P.primeCompl)
    this : IsDomain (Localization (Algebra.algebraMapSubmonoid S P.primeCompl)) := …
    ⊢ Exists fun Q => And Q.IsPrime (Eq (Ideal.comap (algebraMap R S) Q) P)
  -/
  obtain ⟨Qₚ : Ideal Sₚ, Qₚ_maximal⟩ := exists_maximal Sₚ
  /-
    case intro
    R : Type u_1
    inst✝⁵ : CommRing R
    S : Type u_2
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R S
    inst✝² : IsDomain S
    inst✝¹ : Algebra.IsIntegral R S
    P : Ideal R
    inst✝ : P.IsPrime
    hP : LE.le (RingHom.ker (algebraMap R S)) P
    hP0 : Not (Membership.mem (Algebra.algebraMapSubmonoid S P.primeCompl) 0)
    Rₚ : Type u_1 := Localization P.primeCompl
    Sₚ : Type u_2 := Localization (Algebra.algebraMapSubmonoid S P.primeCompl)
    this : IsDomain (Localization (Algebra.algebraMapSubmonoid S P.primeCompl)) := …
    Qₚ : Ideal Sₚ
    Qₚ_maximal : Qₚ.IsMaximal
    ⊢ Exists fun Q => And Q.IsPrime (Eq (Ideal.comap (algebraMap R S) Q) P)
  -/
  let _ : Algebra Rₚ Sₚ := localizationAlgebra P.primeCompl S
  /-
    case intro
    R : Type u_1
    inst✝⁵ : CommRing R
    S : Type u_2
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R S
    inst✝² : IsDomain S
    inst✝¹ : Algebra.IsIntegral R S
    P : Ideal R
    inst✝ : P.IsPrime
    hP : LE.le (RingHom.ker (algebraMap R S)) P
    hP0 : Not (Membership.mem (Algebra.algebraMapSubmonoid S P.primeCompl) 0)
    Rₚ : Type u_1 := Localization P.primeCompl
    Sₚ : Type u_2 := Localization (Algebra.algebraMapSubmonoid S P.primeCompl)
    this : IsDomain (Localization (Algebra.algebraMapSubmonoid S P.primeCompl)) := …
    Qₚ : Ideal Sₚ
    Qₚ_maximal : Qₚ.IsMaximal
    x✝ : Algebra Rₚ Sₚ := localizationAlgebra P.primeCompl S
    ⊢ Exists fun Q => And Q.IsPrime (Eq (Ideal.comap (algebraMap R S) Q) P)
  -/
  have : Algebra.IsIntegral Rₚ Sₚ := ⟨isIntegral_localization⟩
  have Qₚ_max : IsMaximal (comap _ Qₚ) :=
    isMaximal_comap_of_isIntegral_of_isMaximal (R := Rₚ) (S := Sₚ) Qₚ
  /-
    case intro
    R : Type u_1
    inst✝⁵ : CommRing R
    S : Type u_2
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R S
    inst✝² : IsDomain S
    inst✝¹ : Algebra.IsIntegral R S
    P : Ideal R
    inst✝ : P.IsPrime
    hP : LE.le (RingHom.ker (algebraMap R S)) P
    hP0 : Not (Membership.mem (Algebra.algebraMapSubmonoid S P.primeCompl) 0)
    Rₚ : Type u_1 := Localization P.primeCompl
    Sₚ : Type u_2 := Localization (Algebra.algebraMapSubmonoid S P.primeCompl)
    this✝ : IsDomain (Localization (Algebra.algebraMapSubmonoid S P.primeCompl)) : …
    Qₚ : Ideal Sₚ
    Qₚ_maximal : Qₚ.IsMaximal
    x✝ : Algebra Rₚ Sₚ := localizationAlgebra P.primeCompl S
    this : Algebra.IsIntegral Rₚ Sₚ
    Qₚ_max : (Ideal.comap (algebraMap Rₚ Sₚ) Qₚ).IsMaximal
    ⊢ Exists fun Q => And Q.IsPrime (Eq (Ideal.comap (algebraMap R S) Q) P)
  -/
  refine ⟨comap (algebraMap S Sₚ) Qₚ, ⟨comap_isPrime _ Qₚ, ?_⟩⟩
  /-
    case intro
    R : Type u_1
    inst✝⁵ : CommRing R
    S : Type u_2
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R S
    inst✝² : IsDomain S
    inst✝¹ : Algebra.IsIntegral R S
    P : Ideal R
    inst✝ : P.IsPrime
    hP : LE.le (RingHom.ker (algebraMap R S)) P
    hP0 : Not (Membership.mem (Algebra.algebraMapSubmonoid S P.primeCompl) 0)
    Rₚ : Type u_1 := Localization P.primeCompl
    Sₚ : Type u_2 := Localization (Algebra.algebraMapSubmonoid S P.primeCompl)
    this✝ : IsDomain (Localization (Algebra.algebraMapSubmonoid S P.primeCompl)) : …
    Qₚ : Ideal Sₚ
    Qₚ_maximal : Qₚ.IsMaximal
    x✝ : Algebra Rₚ Sₚ := localizationAlgebra P.primeCompl S
    this : Algebra.IsIntegral Rₚ Sₚ
    Qₚ_max : (Ideal.comap (algebraMap Rₚ Sₚ) Qₚ).IsMaximal
    ⊢ Eq (Ideal.comap (algebraMap R S) (Ideal.comap (algebraMap S Sₚ) Qₚ)) P
  -/
  convert Localization.AtPrime.comap_maximalIdeal (I := P)
  rw [comap_comap, ← IsLocalRing.eq_maximalIdeal Qₚ_max,
    ← IsLocalization.map_comp (P := S) (Q := Sₚ) (g := algebraMap R S)
    (M := P.primeCompl) (T := Algebra.algebraMapSubmonoid S P.primeCompl) (S := Rₚ)
    (fun p hp => Algebra.mem_algebraMapSubmonoid_of_mem ⟨p, hp⟩) ]
  /-
    case h.e'_2
    R : Type u_1
    inst✝⁵ : CommRing R
    S : Type u_2
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R S
    inst✝² : IsDomain S
    inst✝¹ : Algebra.IsIntegral R S
    P : Ideal R
    inst✝ : P.IsPrime
    hP : LE.le (RingHom.ker (algebraMap R S)) P
    hP0 : Not (Membership.mem (Algebra.algebraMapSubmonoid S P.primeCompl) 0)
    Rₚ : Type u_1 := Localization P.primeCompl
    Sₚ : Type u_2 := Localization (Algebra.algebraMapSubmonoid S P.primeCompl)
    this✝ : IsDomain (Localization (Algebra.algebraMapSubmonoid S P.primeCompl)) : …
    Qₚ : Ideal Sₚ
    Qₚ_maximal : Qₚ.IsMaximal
    x✝ : Algebra Rₚ Sₚ := localizationAlgebra P.primeCompl S
    this : Algebra.IsIntegral Rₚ Sₚ
    Qₚ_max : (Ideal.comap (algebraMap Rₚ Sₚ) Qₚ).IsMaximal
    ⊢ Eq (Ideal.comap ((IsLocalization.map Sₚ (algebraMap R S) ⋯).comp (algebraMap …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- More general going-up theorem than `exists_ideal_over_prime_of_isIntegral_of_isDomain`.
TODO: Version of going-up theorem with arbitrary length chains (by induction on this)?
  Not sure how best to write an ascending chain in Lean -/
theorem exists_ideal_over_prime_of_isIntegral_of_isPrime
    [Algebra.IsIntegral R S] (P : Ideal R) [IsPrime P]
    (I : Ideal S) [IsPrime I] (hIP : I.comap (algebraMap R S) ≤ P) :
    ∃ Q ≥ I, IsPrime Q ∧ Q.comap (algebraMap R S) = P := by
  obtain ⟨Q' : Ideal (S ⧸ I), ⟨Q'_prime, hQ'⟩⟩ :=
    @exists_ideal_over_prime_of_isIntegral_of_isDomain (R ⧸ I.comap (algebraMap R S)) _ (S ⧸ I) _
      Ideal.quotientAlgebra _ _
      (map (Ideal.Quotient.mk (I.comap (algebraMap R S))) P)
      (map_isPrime_of_surjective Quotient.mk_surjective (by simp [hIP]))
      (le_trans (le_of_eq ((RingHom.injective_iff_ker_eq_bot _).1 algebraMap_quotient_injective))
        bot_le)
  /-
    case intro.intro
    R : Type u_1
    inst✝⁵ : CommRing R
    S : Type u_2
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R S
    inst✝² : Algebra.IsIntegral R S
    P : Ideal R
    inst✝¹ : P.IsPrime
    I : Ideal S
    inst✝ : I.IsPrime
    hIP : LE.le (Ideal.comap (algebraMap R S) I) P
    Q' : Ideal (HasQuotient.Quotient S I)
    Q'_prime : Q'.IsPrime
    hQ' : Eq (Ideal.comap (algebraMap (HasQuotient.Quotient R (Ideal.comap (algebr …
    ⊢ Exists fun Q => And (GE.ge Q I) (And Q.IsPrime (Eq (Ideal.comap (algebraMap  …
  -/
  refine ⟨Q'.comap _, le_trans (le_of_eq mk_ker.symm) (ker_le_comap _), ⟨comap_isPrime _ Q', ?_⟩⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝⁵ : CommRing R
    S : Type u_2
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R S
    inst✝² : Algebra.IsIntegral R S
    P : Ideal R
    inst✝¹ : P.IsPrime
    I : Ideal S
    inst✝ : I.IsPrime
    hIP : LE.le (Ideal.comap (algebraMap R S) I) P
    Q' : Ideal (HasQuotient.Quotient S I)
    Q'_prime : Q'.IsPrime
    hQ' : Eq (Ideal.comap (algebraMap (HasQuotient.Quotient R (Ideal.comap (algebr …
    ⊢ Eq (Ideal.comap (algebraMap R S) (Ideal.comap (Ideal.Quotient.mk I) Q')) P
  -/
  rw [comap_comap]
  refine _root_.trans ?_ (_root_.trans (congr_arg (comap (Ideal.Quotient.mk
    (comap (algebraMap R S) I))) hQ') ?_)
    /-
      case intro.intro.refine_1
      R : Type u_1
      inst✝⁵ : CommRing R
      S : Type u_2
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R S
      inst✝² : Algebra.IsIntegral R S
      P : Ideal R
      inst✝¹ : P.IsPrime
      I : Ideal S
      inst✝ : I.IsPrime
      hIP : LE.le (Ideal.comap (algebraMap R S) I) P
      Q' : Ideal (HasQuotient.Quotient S I)
      Q'_prime : Q'.IsPrime
      hQ' : Eq (Ideal.comap (algebraMap (HasQuotient.Quotient R (Ideal.comap (algebr …
      ⊢ Eq (Ideal.comap ((Ideal.Quotient.mk I).comp (algebraMap R S)) Q') (Ideal.com …
    -/
  · rw [comap_comap]
    /-
      case intro.intro.refine_1
      R : Type u_1
      inst✝⁵ : CommRing R
      S : Type u_2
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R S
      inst✝² : Algebra.IsIntegral R S
      P : Ideal R
      inst✝¹ : P.IsPrime
      I : Ideal S
      inst✝ : I.IsPrime
      hIP : LE.le (Ideal.comap (algebraMap R S) I) P
      Q' : Ideal (HasQuotient.Quotient S I)
      Q'_prime : Q'.IsPrime
      hQ' : Eq (Ideal.comap (algebraMap (HasQuotient.Quotient R (Ideal.comap (algebr …
      ⊢ Eq (Ideal.comap ((Ideal.Quotient.mk I).comp (algebraMap R S)) Q') (Ideal.com …
    -/
    exact congr_arg (comap · Q') (RingHom.ext fun r => rfl)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      R : Type u_1
      inst✝⁵ : CommRing R
      S : Type u_2
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R S
      inst✝² : Algebra.IsIntegral R S
      P : Ideal R
      inst✝¹ : P.IsPrime
      I : Ideal S
      inst✝ : I.IsPrime
      hIP : LE.le (Ideal.comap (algebraMap R S) I) P
      Q' : Ideal (HasQuotient.Quotient S I)
      Q'_prime : Q'.IsPrime
      hQ' : Eq (Ideal.comap (algebraMap (HasQuotient.Quotient R (Ideal.comap (algebr …
      ⊢ Eq (Ideal.comap (Ideal.Quotient.mk (Ideal.comap (algebraMap R S) I)) (Ideal. …
    -/
  · refine _root_.trans (comap_map_of_surjective _ Quotient.mk_surjective _) (sup_eq_left.2 ?_)
    /-
      case intro.intro.refine_2
      R : Type u_1
      inst✝⁵ : CommRing R
      S : Type u_2
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R S
      inst✝² : Algebra.IsIntegral R S
      P : Ideal R
      inst✝¹ : P.IsPrime
      I : Ideal S
      inst✝ : I.IsPrime
      hIP : LE.le (Ideal.comap (algebraMap R S) I) P
      Q' : Ideal (HasQuotient.Quotient S I)
      Q'_prime : Q'.IsPrime
      hQ' : Eq (Ideal.comap (algebraMap (HasQuotient.Quotient R (Ideal.comap (algebr …
      ⊢ LE.le (Ideal.comap (Ideal.Quotient.mk (Ideal.comap (algebraMap R S) I)) Bot. …
    -/
    simpa [← RingHom.ker_eq_comap_bot] using hIP
    /-
      🎉 no goals
    -/


lemma exists_ideal_comap_le_prime (P : Ideal R) [P.IsPrime]
    (I : Ideal S) (hI : I.comap (algebraMap R S) ≤ P) :
    ∃ Q ≥ I, Q.IsPrime ∧ Q.comap (algebraMap R S) ≤ P := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    P : Ideal R
    inst✝ : P.IsPrime
    I : Ideal S
    hI : LE.le (Ideal.comap (algebraMap R S) I) P
    ⊢ Exists fun Q => And (GE.ge Q I) (And Q.IsPrime (LE.le (Ideal.comap (algebraM …
  -/
  let Sₚ := Localization (Algebra.algebraMapSubmonoid S P.primeCompl)
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    P : Ideal R
    inst✝ : P.IsPrime
    I : Ideal S
    hI : LE.le (Ideal.comap (algebraMap R S) I) P
    Sₚ : Type u_2 := Localization (Algebra.algebraMapSubmonoid S P.primeCompl)
    ⊢ Exists fun Q => And (GE.ge Q I) (And Q.IsPrime (LE.le (Ideal.comap (algebraM …
  -/
  let Iₚ := I.map (algebraMap S Sₚ)
  have hI' : Disjoint (Algebra.algebraMapSubmonoid S P.primeCompl : Set S) I := by
    rw [Set.disjoint_iff]
    rintro _ ⟨⟨x, hx : x ∉ P, rfl⟩, hx'⟩
    exact (hx (hI hx')).elim
  have : Iₚ ≠ ⊤ := by
    rw [Ne, Ideal.eq_top_iff_one, IsLocalization.mem_map_algebraMap_iff
      (Algebra.algebraMapSubmonoid S P.primeCompl) Sₚ, not_exists]
    simp only [one_mul, IsLocalization.eq_iff_exists (Algebra.algebraMapSubmonoid S P.primeCompl),
      not_exists]
    exact fun x c ↦ hI'.ne_of_mem (mul_mem c.2 x.2.2) (I.mul_mem_left c x.1.2)
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    P : Ideal R
    inst✝ : P.IsPrime
    I : Ideal S
    hI : LE.le (Ideal.comap (algebraMap R S) I) P
    Sₚ : Type u_2 := Localization (Algebra.algebraMapSubmonoid S P.primeCompl)
    Iₚ : Ideal Sₚ := Ideal.map (algebraMap S Sₚ) I
    hI' : Disjoint ↑(Algebra.algebraMapSubmonoid S P.primeCompl) ↑I
    this : Ne Iₚ Top.top
    ⊢ Exists fun Q => And (GE.ge Q I) (And Q.IsPrime (LE.le (Ideal.comap (algebraM …
  -/
  obtain ⟨M, hM, hM'⟩ := Ideal.exists_le_maximal _ this
  /-
    case intro.intro
    R : Type u_1
    inst✝³ : CommRing R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    P : Ideal R
    inst✝ : P.IsPrime
    I : Ideal S
    hI : LE.le (Ideal.comap (algebraMap R S) I) P
    Sₚ : Type u_2 := Localization (Algebra.algebraMapSubmonoid S P.primeCompl)
    Iₚ : Ideal Sₚ := Ideal.map (algebraMap S Sₚ) I
    hI' : Disjoint ↑(Algebra.algebraMapSubmonoid S P.primeCompl) ↑I
    this : Ne Iₚ Top.top
    M : Ideal Sₚ
    hM : M.IsMaximal
    hM' : LE.le Iₚ M
    ⊢ Exists fun Q => And (GE.ge Q I) (And Q.IsPrime (LE.le (Ideal.comap (algebraM …
  -/
  refine ⟨_, Ideal.map_le_iff_le_comap.mp hM', hM.isPrime.comap _, ?_⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝³ : CommRing R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    P : Ideal R
    inst✝ : P.IsPrime
    I : Ideal S
    hI : LE.le (Ideal.comap (algebraMap R S) I) P
    Sₚ : Type u_2 := Localization (Algebra.algebraMapSubmonoid S P.primeCompl)
    Iₚ : Ideal Sₚ := Ideal.map (algebraMap S Sₚ) I
    hI' : Disjoint ↑(Algebra.algebraMapSubmonoid S P.primeCompl) ↑I
    this : Ne Iₚ Top.top
    M : Ideal Sₚ
    hM : M.IsMaximal
    hM' : LE.le Iₚ M
    ⊢ LE.le (Ideal.comap (algebraMap R S) (Ideal.comap (algebraMap S Sₚ) M)) P
  -/
  intro x hx
  /-
    case intro.intro
    R : Type u_1
    inst✝³ : CommRing R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    P : Ideal R
    inst✝ : P.IsPrime
    I : Ideal S
    hI : LE.le (Ideal.comap (algebraMap R S) I) P
    Sₚ : Type u_2 := Localization (Algebra.algebraMapSubmonoid S P.primeCompl)
    Iₚ : Ideal Sₚ := Ideal.map (algebraMap S Sₚ) I
    hI' : Disjoint ↑(Algebra.algebraMapSubmonoid S P.primeCompl) ↑I
    this : Ne Iₚ Top.top
    M : Ideal Sₚ
    hM : M.IsMaximal
    hM' : LE.le Iₚ M
    x : R
    hx : Membership.mem (Ideal.comap (algebraMap R S) (Ideal.comap (algebraMap S S …
    ⊢ Membership.mem P x
  -/
  by_contra hx'
  exact Set.disjoint_left.mp ((IsLocalization.isPrime_iff_isPrime_disjoint
    (Algebra.algebraMapSubmonoid S P.primeCompl) Sₚ M).mp hM.isPrime).2 ⟨_, hx', rfl⟩ hx


theorem exists_ideal_over_prime_of_isIntegral [Algebra.IsIntegral R S] (P : Ideal R) [IsPrime P]
    (I : Ideal S) (hIP : I.comap (algebraMap R S) ≤ P) :
    ∃ Q ≥ I, IsPrime Q ∧ Q.comap (algebraMap R S) = P := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    S : Type u_2
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : Algebra.IsIntegral R S
    P : Ideal R
    inst✝ : P.IsPrime
    I : Ideal S
    hIP : LE.le (Ideal.comap (algebraMap R S) I) P
    ⊢ Exists fun Q => And (GE.ge Q I) (And Q.IsPrime (Eq (Ideal.comap (algebraMap  …
  -/
  have ⟨P', hP, hP', hP''⟩ := exists_ideal_comap_le_prime P I hIP
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    S : Type u_2
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : Algebra.IsIntegral R S
    P : Ideal R
    inst✝ : P.IsPrime
    I : Ideal S
    hIP : LE.le (Ideal.comap (algebraMap R S) I) P
    P' : Ideal S
    hP : GE.ge P' I
    hP' : P'.IsPrime
    hP'' : LE.le (Ideal.comap (algebraMap R S) P') P
    ⊢ Exists fun Q => And (GE.ge Q I) (And Q.IsPrime (Eq (Ideal.comap (algebraMap  …
  -/
  obtain ⟨Q, hQ, hQ', hQ''⟩ := exists_ideal_over_prime_of_isIntegral_of_isPrime P P' hP''
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝⁴ : CommRing R
    S : Type u_2
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : Algebra.IsIntegral R S
    P : Ideal R
    inst✝ : P.IsPrime
    I : Ideal S
    hIP : LE.le (Ideal.comap (algebraMap R S) I) P
    P' : Ideal S
    hP : GE.ge P' I
    hP' : P'.IsPrime
    hP'' : LE.le (Ideal.comap (algebraMap R S) P') P
    Q : Ideal S
    hQ : GE.ge Q P'
    hQ' : Q.IsPrime
    hQ'' : Eq (Ideal.comap (algebraMap R S) Q) P
    ⊢ Exists fun Q => And (GE.ge Q I) (And Q.IsPrime (Eq (Ideal.comap (algebraMap  …
  -/
  exact ⟨Q, hP.trans hQ, hQ', hQ''⟩
  /-
    🎉 no goals
  -/


/-- `comap (algebraMap R S)` is a surjection from the max spec of `S` to max spec of `R`.
`hP : (algebraMap R S).ker ≤ P` is a slight generalization of the extension being injective -/
theorem exists_ideal_over_maximal_of_isIntegral [Algebra.IsIntegral R S]
    (P : Ideal R) [P_max : IsMaximal P] (hP : RingHom.ker (algebraMap R S) ≤ P) :
    ∃ Q : Ideal S, IsMaximal Q ∧ Q.comap (algebraMap R S) = P := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : Algebra.IsIntegral R S
    P : Ideal R
    P_max : P.IsMaximal
    hP : LE.le (RingHom.ker (algebraMap R S)) P
    ⊢ Exists fun Q => And Q.IsMaximal (Eq (Ideal.comap (algebraMap R S) Q) P)
  -/
  obtain ⟨Q, -, Q_prime, hQ⟩ := exists_ideal_over_prime_of_isIntegral P ⊥ hP
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝³ : CommRing R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : Algebra.IsIntegral R S
    P : Ideal R
    P_max : P.IsMaximal
    hP : LE.le (RingHom.ker (algebraMap R S)) P
    Q : Ideal S
    Q_prime : Q.IsPrime
    hQ : Eq (Ideal.comap (algebraMap R S) Q) P
    ⊢ Exists fun Q => And Q.IsMaximal (Eq (Ideal.comap (algebraMap R S) Q) P)
  -/
  exact ⟨Q, isMaximal_of_isIntegral_of_isMaximal_comap _ (hQ.symm ▸ P_max), hQ⟩
  /-
    🎉 no goals
  -/


lemma map_eq_top_iff_of_ker_le {R S} [CommRing R] [CommRing S]
    (f : R →+* S) {I : Ideal R} (hf₁ : RingHom.ker f ≤ I) (hf₂ : f.IsIntegral) :
    I.map f = ⊤ ↔ I = ⊤ := by
  /-
    R : Type u_3
    S : Type u_4
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    I : Ideal R
    hf₁ : LE.le (RingHom.ker f) I
    hf₂ : f.IsIntegral
    ⊢ Iff (Eq (Ideal.map f I) Top.top) (Eq I Top.top)
  -/
  constructor; swap
    /-
      case mpr
      R : Type u_3
      S : Type u_4
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      I : Ideal R
      hf₁ : LE.le (RingHom.ker f) I
      hf₂ : f.IsIntegral
      ⊢ Eq I Top.top → Eq (Ideal.map f I) Top.top
    -/
  · rintro rfl; exact Ideal.map_top _
                /-
                  🎉 no goals
                -/
  /-
    case mp
    R : Type u_3
    S : Type u_4
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    I : Ideal R
    hf₁ : LE.le (RingHom.ker f) I
    hf₂ : f.IsIntegral
    ⊢ Eq (Ideal.map f I) Top.top → Eq I Top.top
  -/
  contrapose
  /-
    case mp
    R : Type u_3
    S : Type u_4
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    I : Ideal R
    hf₁ : LE.le (RingHom.ker f) I
    hf₂ : f.IsIntegral
    ⊢ Not (Eq I Top.top) → Not (Eq (Ideal.map f I) Top.top)
  -/
  intro h
  /-
    case mp
    R : Type u_3
    S : Type u_4
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    I : Ideal R
    hf₁ : LE.le (RingHom.ker f) I
    hf₂ : f.IsIntegral
    h : Not (Eq I Top.top)
    ⊢ Not (Eq (Ideal.map f I) Top.top)
  -/
  obtain ⟨m, _, hm⟩ := Ideal.exists_le_maximal I h
  /-
    case mp.intro.intro
    R : Type u_3
    S : Type u_4
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    I : Ideal R
    hf₁ : LE.le (RingHom.ker f) I
    hf₂ : f.IsIntegral
    h : Not (Eq I Top.top)
    m : Ideal R
    left✝ : m.IsMaximal
    hm : LE.le I m
    ⊢ Not (Eq (Ideal.map f I) Top.top)
  -/
  let _ := f.toAlgebra
  /-
    case mp.intro.intro
    R : Type u_3
    S : Type u_4
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    I : Ideal R
    hf₁ : LE.le (RingHom.ker f) I
    hf₂ : f.IsIntegral
    h : Not (Eq I Top.top)
    m : Ideal R
    left✝ : m.IsMaximal
    hm : LE.le I m
    x✝ : Algebra R S := f.toAlgebra
    ⊢ Not (Eq (Ideal.map f I) Top.top)
  -/
  have : Algebra.IsIntegral _ _ := ⟨hf₂⟩
  /-
    case mp.intro.intro
    R : Type u_3
    S : Type u_4
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    I : Ideal R
    hf₁ : LE.le (RingHom.ker f) I
    hf₂ : f.IsIntegral
    h : Not (Eq I Top.top)
    m : Ideal R
    left✝ : m.IsMaximal
    hm : LE.le I m
    x✝ : Algebra R S := f.toAlgebra
    this : Algebra.IsIntegral R S
    ⊢ Not (Eq (Ideal.map f I) Top.top)
  -/
  obtain ⟨m', _, rfl⟩ := exists_ideal_over_maximal_of_isIntegral m (hf₁.trans hm)
  /-
    case mp.intro.intro.intro.intro
    R : Type u_3
    S : Type u_4
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    I : Ideal R
    hf₁ : LE.le (RingHom.ker f) I
    hf₂ : f.IsIntegral
    h : Not (Eq I Top.top)
    x✝ : Algebra R S := f.toAlgebra
    this : Algebra.IsIntegral R S
    m' : Ideal S
    left✝¹ : m'.IsMaximal
    left✝ : (Ideal.comap (algebraMap R S) m').IsMaximal
    hm : LE.le I (Ideal.comap (algebraMap R S) m')
    ⊢ Not (Eq (Ideal.map f I) Top.top)
  -/
  rw [← map_le_iff_le_comap] at hm
  /-
    case mp.intro.intro.intro.intro
    R : Type u_3
    S : Type u_4
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    I : Ideal R
    hf₁ : LE.le (RingHom.ker f) I
    hf₂ : f.IsIntegral
    h : Not (Eq I Top.top)
    x✝ : Algebra R S := f.toAlgebra
    this : Algebra.IsIntegral R S
    m' : Ideal S
    left✝¹ : m'.IsMaximal
    left✝ : (Ideal.comap (algebraMap R S) m').IsMaximal
    hm : LE.le (Ideal.map (algebraMap R S) I) m'
    ⊢ Not (Eq (Ideal.map f I) Top.top)
  -/
  exact (hm.trans_lt (lt_top_iff_ne_top.mpr (IsMaximal.ne_top ‹_›))).ne
  /-
    🎉 no goals
  -/


lemma map_eq_top_iff {R S} [CommRing R] [CommRing S]
    (f : R →+* S) {I : Ideal R} (hf₁ : Function.Injective f) (hf₂ : f.IsIntegral) :
    I.map f = ⊤ ↔ I = ⊤ :=
                                 /-
                                   R : Type u_3
                                   S : Type u_4
                                   inst✝¹ : CommRing R
                                   inst✝ : CommRing S
                                   f : RingHom R S
                                   I : Ideal R
                                   hf₁ : Function.Injective ⇑f
                                   hf₂ : f.IsIntegral
                                   ⊢ LE.le (RingHom.ker f) I
                                 -/
  map_eq_top_iff_of_ker_le f (by simp [f.injective_iff_ker_eq_bot.mp hf₁]) hf₂
                                 /-
                                   🎉 no goals
                                 -/


/-- The ideal obtained by pulling back the ideal `P` from `B` to `A`. -/
abbrev under : Ideal A := Ideal.comap (algebraMap A B) P


theorem under_def : P.under A = Ideal.comap (algebraMap A B) P := rfl


instance IsPrime.under [hP : P.IsPrime] : (P.under A).IsPrime :=
  hP.comap (algebraMap A B)


@[simp]
lemma under_smul {G : Type*} [Group G] [MulSemiringAction G B] [SMulCommClass G A B] (g : G) :
    (g • P : Ideal B).under A = P.under A := by
  /-
    A : Type u_2
    inst✝⁵ : CommSemiring A
    B : Type u_3
    inst✝⁴ : Semiring B
    inst✝³ : Algebra A B
    P : Ideal B
    G : Type u_5
    inst✝² : Group G
    inst✝¹ : MulSemiringAction G B
    inst✝ : SMulCommClass G A B
    g : G
    ⊢ Eq (Ideal.under A (HSMul.hSMul g P)) (Ideal.under A P)
  -/
  ext a
  /-
    case h
    A : Type u_2
    inst✝⁵ : CommSemiring A
    B : Type u_3
    inst✝⁴ : Semiring B
    inst✝³ : Algebra A B
    P : Ideal B
    G : Type u_5
    inst✝² : Group G
    inst✝¹ : MulSemiringAction G B
    inst✝ : SMulCommClass G A B
    g : G
    a : A
    ⊢ Iff (Membership.mem (Ideal.under A (HSMul.hSMul g P)) a) (Membership.mem (Id …
  -/
  rw [mem_comap, mem_comap, mem_pointwise_smul_iff_inv_smul_mem, smul_algebraMap]
  /-
    🎉 no goals
  -/


variable (B) in
theorem under_top : under A (⊤ : Ideal B) = ⊤ := comap_top


/-- `P` lies over `p` if `p` is the preimage of `P` of the `algebraMap`. -/
class LiesOver : Prop where
  over : p = P.under A


instance over_under : P.LiesOver (P.under A) where over := rfl


theorem over_def [P.LiesOver p] : p = P.under A := LiesOver.over


theorem mem_of_liesOver [P.LiesOver p] (x : A) : x ∈ p ↔ algebraMap A B x ∈ P := by
  /-
    A : Type u_2
    inst✝³ : CommSemiring A
    B : Type u_3
    inst✝² : Semiring B
    inst✝¹ : Algebra A B
    P : Ideal B
    p : Ideal A
    inst✝ : P.LiesOver p
    x : A
    ⊢ Iff (Membership.mem p x) (Membership.mem P ((algebraMap A B) x))
  -/
  rw [P.over_def p]
  /-
    A : Type u_2
    inst✝³ : CommSemiring A
    B : Type u_3
    inst✝² : Semiring B
    inst✝¹ : Algebra A B
    P : Ideal B
    p : Ideal A
    inst✝ : P.LiesOver p
    x : A
    ⊢ Iff (Membership.mem (Ideal.under A P) x) (Membership.mem P ((algebraMap A B) …
  -/
  rfl
  /-
    🎉 no goals
  -/


variable (A B) in
instance top_liesOver_top : (⊤ : Ideal B).LiesOver (⊤ : Ideal A) where
  over := (under_top A B).symm


theorem eq_top_iff_of_liesOver [P.LiesOver p] : P = ⊤ ↔ p = ⊤ := by
  /-
    A : Type u_2
    inst✝³ : CommSemiring A
    B : Type u_3
    inst✝² : Semiring B
    inst✝¹ : Algebra A B
    P : Ideal B
    p : Ideal A
    inst✝ : P.LiesOver p
    ⊢ Iff (Eq P Top.top) (Eq p Top.top)
  -/
  rw [P.over_def p]
  /-
    A : Type u_2
    inst✝³ : CommSemiring A
    B : Type u_3
    inst✝² : Semiring B
    inst✝¹ : Algebra A B
    P : Ideal B
    p : Ideal A
    inst✝ : P.LiesOver p
    ⊢ Iff (Eq P Top.top) (Eq (Ideal.under A P) Top.top)
  -/
  exact comap_eq_top_iff.symm
  /-
    🎉 no goals
  -/


theorem LiesOver.of_eq_comap [Q.LiesOver p] {F : Type*} [FunLike F B C]
    [AlgHomClass F A B C] (f : F) (h : P = Q.comap f) : P.LiesOver p where
  over := by
    /-
      A : Type u_2
      inst✝⁷ : CommSemiring A
      B : Type u_3
      C : Type u_4
      inst✝⁶ : Semiring B
      inst✝⁵ : Semiring C
      inst✝⁴ : Algebra A B
      inst✝³ : Algebra A C
      P : Ideal B
      Q : Ideal C
      p : Ideal A
      inst✝² : Q.LiesOver p
      F : Type u_5
      inst✝¹ : FunLike F B C
      inst✝ : AlgHomClass F A B C
      f : F
      h : Eq P (Ideal.comap f Q)
      ⊢ Eq p (Ideal.under A P)
    -/
    rw [h]
    exact (over_def Q p).trans <|
      congrFun (congrFun (congrArg comap ((f : B →ₐ[A] C).comp_algebraMap.symm)) _) Q


theorem LiesOver.of_eq_map_equiv [P.LiesOver p] {E : Type*} [EquivLike E B C]
    [AlgEquivClass E A B C] (σ : E) (h : Q = P.map σ) : Q.LiesOver p := by
  /-
    A : Type u_2
    inst✝⁷ : CommSemiring A
    B : Type u_3
    C : Type u_4
    inst✝⁶ : Semiring B
    inst✝⁵ : Semiring C
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra A C
    P : Ideal B
    Q : Ideal C
    p : Ideal A
    inst✝² : P.LiesOver p
    E : Type u_5
    inst✝¹ : EquivLike E B C
    inst✝ : AlgEquivClass E A B C
    σ : E
    h : Eq Q (Ideal.map σ P)
    ⊢ Q.LiesOver p
  -/
  rw [← show _ = P.map σ from comap_symm (σ : B ≃+* C)] at h
  /-
    A : Type u_2
    inst✝⁷ : CommSemiring A
    B : Type u_3
    C : Type u_4
    inst✝⁶ : Semiring B
    inst✝⁵ : Semiring C
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra A C
    P : Ideal B
    Q : Ideal C
    p : Ideal A
    inst✝² : P.LiesOver p
    E : Type u_5
    inst✝¹ : EquivLike E B C
    inst✝ : AlgEquivClass E A B C
    σ : E
    h : Eq Q (Ideal.comap (↑σ).symm P)
    ⊢ Q.LiesOver p
  -/
  exact of_eq_comap p (σ : B ≃ₐ[A] C).symm h
  /-
    🎉 no goals
  -/


instance comap_liesOver [Q.LiesOver p] {F : Type*} [FunLike F B C] [AlgHomClass F A B C]
    (f : F) : (Q.comap f).LiesOver p :=
  LiesOver.of_eq_comap p f rfl


instance map_equiv_liesOver [P.LiesOver p] {E : Type*} [EquivLike E B C] [AlgEquivClass E A B C]
    (σ : E) : (P.map σ).LiesOver p :=
  LiesOver.of_eq_map_equiv p σ rfl


@[simp]
theorem under_under : (𝔓.under B).under A  = 𝔓.under A := by
  /-
    A : Type u_2
    inst✝⁶ : CommSemiring A
    B : Type u_3
    inst✝⁵ : CommSemiring B
    C : Type u_4
    inst✝⁴ : Semiring C
    inst✝³ : Algebra A B
    inst✝² : Algebra B C
    inst✝¹ : Algebra A C
    inst✝ : IsScalarTower A B C
    𝔓 : Ideal C
    ⊢ Eq (Ideal.under A (Ideal.under B 𝔓)) (Ideal.under A 𝔓)
  -/
  simp_rw [comap_comap, ← IsScalarTower.algebraMap_eq]
  /-
    🎉 no goals
  -/


theorem LiesOver.trans [𝔓.LiesOver P] [P.LiesOver p] : 𝔓.LiesOver p where
             /-
               A : Type u_2
               inst✝⁸ : CommSemiring A
               B : Type u_3
               inst✝⁷ : CommSemiring B
               C : Type u_4
               inst✝⁶ : Semiring C
               inst✝⁵ : Algebra A B
               inst✝⁴ : Algebra B C
               inst✝³ : Algebra A C
               inst✝² : IsScalarTower A B C
               𝔓 : Ideal C
               P : Ideal B
               p : Ideal A
               inst✝¹ : 𝔓.LiesOver P
               inst✝ : P.LiesOver p
               ⊢ Eq p (Ideal.under A 𝔓)
             -/
  over := by rw [P.over_def p, 𝔓.over_def P, under_under]
             /-
               🎉 no goals
             -/


theorem LiesOver.tower_bot [hp : 𝔓.LiesOver p] [hP : 𝔓.LiesOver P] : P.LiesOver p where
             /-
               A : Type u_2
               inst✝⁶ : CommSemiring A
               B : Type u_3
               inst✝⁵ : CommSemiring B
               C : Type u_4
               inst✝⁴ : Semiring C
               inst✝³ : Algebra A B
               inst✝² : Algebra B C
               inst✝¹ : Algebra A C
               inst✝ : IsScalarTower A B C
               𝔓 : Ideal C
               P : Ideal B
               p : Ideal A
               hp : 𝔓.LiesOver p
               hP : 𝔓.LiesOver P
               ⊢ Eq p (Ideal.under A P)
             -/
  over := by rw [𝔓.over_def p, 𝔓.over_def P, under_under]
             /-
               🎉 no goals
             -/


instance under_liesOver_of_liesOver [𝔓.LiesOver p] : (𝔓.under B).LiesOver p :=
  LiesOver.tower_bot 𝔓 (𝔓.under B) p


@[simp]
theorem under_bot : under A (⊥ : Ideal B) = ⊥ :=
  comap_bot_of_injective (algebraMap A B) (NoZeroSMulDivisors.algebraMap_injective A B)


instance bot_liesOver_bot : (⊥ : Ideal B).LiesOver (⊥ : Ideal A) where
  over := (under_bot A B).symm


variable {A B} in
theorem ne_bot_of_liesOver_of_ne_bot (hp : p ≠ ⊥) (P : Ideal B) [P.LiesOver p] : P ≠ ⊥ := by
  /-
    A : Type u_2
    inst✝⁵ : CommRing A
    B : Type u_3
    inst✝⁴ : Ring B
    inst✝³ : Nontrivial B
    inst✝² : Algebra A B
    inst✝¹ : NoZeroSMulDivisors A B
    p : Ideal A
    hp : Ne p Bot.bot
    P : Ideal B
    inst✝ : P.LiesOver p
    ⊢ Ne P Bot.bot
  -/
  contrapose! hp
  /-
    A : Type u_2
    inst✝⁵ : CommRing A
    B : Type u_3
    inst✝⁴ : Ring B
    inst✝³ : Nontrivial B
    inst✝² : Algebra A B
    inst✝¹ : NoZeroSMulDivisors A B
    p : Ideal A
    P : Ideal B
    inst✝ : P.LiesOver p
    hp : Eq P Bot.bot
    ⊢ Eq p Bot.bot
  -/
  rw [over_def P p, hp, under_bot]
  /-
    🎉 no goals
  -/


/-- If `P` lies over `p`, then canonically `B ⧸ P` is a `A ⧸ p`-algebra. -/
instance algebraOfLiesOver : Algebra (A ⧸ p) (B ⧸ P) :=
  algebraQuotientOfLEComap (le_of_eq (P.over_def p))


instance isScalarTower_of_liesOver : IsScalarTower R (A ⧸ p) (B ⧸ P) :=
  IsScalarTower.of_algebraMap_eq' <|
    congrArg (algebraMap B (B ⧸ P)).comp (IsScalarTower.algebraMap_eq R A B)


/-- `B ⧸ P` is a finite `A ⧸ p`-module if `B` is a finite `A`-module. -/
instance module_finite_of_liesOver [Module.Finite A B] : Module.Finite (A ⧸ p) (B ⧸ P) :=
  Module.Finite.of_restrictScalars_finite A (A ⧸ p) (B ⧸ P)


/-- `B ⧸ P` is a finitely generated `A ⧸ p`-algebra if `B` is a finitely generated `A`-algebra. -/
instance algebra_finiteType_of_liesOver [Algebra.FiniteType A B] :
    Algebra.FiniteType (A ⧸ p) (B ⧸ P) :=
  Algebra.FiniteType.of_restrictScalars_finiteType A (A ⧸ p) (B ⧸ P)


/-- `B ⧸ P` is a Noetherian `A ⧸ p`-module if `B` is a Noetherian `A`-module. -/
instance isNoetherian_of_liesOver [IsNoetherian A B] : IsNoetherian (A ⧸ p) (B ⧸ P) :=
  isNoetherian_of_tower A inferInstance


theorem algebraMap_injective_of_liesOver : Function.Injective (algebraMap (A ⧸ p) (B ⧸ P)) := by
  /-
    A : Type u_3
    B : Type u_4
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    P : Ideal B
    p : Ideal A
    inst✝ : P.LiesOver p
    ⊢ Function.Injective ⇑(algebraMap (HasQuotient.Quotient A p) (HasQuotient.Quot …
  -/
  rintro ⟨a⟩ ⟨b⟩ hab
  /-
    case mk.mk
    A : Type u_3
    B : Type u_4
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    P : Ideal B
    p : Ideal A
    inst✝ : P.LiesOver p
    a₁✝ : HasQuotient.Quotient A p
    a : A
    a₂✝ : HasQuotient.Quotient A p
    b : A
    hab : Eq ((algebraMap (HasQuotient.Quotient A p) (HasQuotient.Quotient B P)) ( …
    ⊢ Eq (Quot.mk (⇑(Submodule.quotientRel p)) a) (Quot.mk (⇑(Submodule.quotientRe …
  -/
  apply Quotient.eq.mpr ((mem_of_liesOver P p (a - b)).mpr _)
  /-
    A : Type u_3
    B : Type u_4
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    P : Ideal B
    p : Ideal A
    inst✝ : P.LiesOver p
    a₁✝ : HasQuotient.Quotient A p
    a : A
    a₂✝ : HasQuotient.Quotient A p
    b : A
    hab : Eq ((algebraMap (HasQuotient.Quotient A p) (HasQuotient.Quotient B P)) ( …
    ⊢ Membership.mem P ((algebraMap A B) (HSub.hSub a b))
  -/
  rw [RingHom.map_sub]
  /-
    A : Type u_3
    B : Type u_4
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    P : Ideal B
    p : Ideal A
    inst✝ : P.LiesOver p
    a₁✝ : HasQuotient.Quotient A p
    a : A
    a₂✝ : HasQuotient.Quotient A p
    b : A
    hab : Eq ((algebraMap (HasQuotient.Quotient A p) (HasQuotient.Quotient B P)) ( …
    ⊢ Membership.mem P (HSub.hSub ((algebraMap A B) a) ((algebraMap A B) b))
  -/
  exact Quotient.eq.mp hab
  /-
    🎉 no goals
  -/


instance [P.IsPrime] : NoZeroSMulDivisors (A ⧸ p) (B ⧸ P) :=
  NoZeroSMulDivisors.of_algebraMap_injective (algebraMap_injective_of_liesOver P p)


variable {p} in
theorem nontrivial_of_liesOver_of_ne_top (hp : p ≠ ⊤) : Nontrivial (B ⧸ P) :=
  Quotient.nontrivial ((eq_top_iff_of_liesOver P p).mp.mt hp)


theorem nontrivial_of_liesOver_of_isPrime [hp : p.IsPrime] : Nontrivial (B ⧸ P) :=
  nontrivial_of_liesOver_of_ne_top P hp.ne_top


/-- An `A ⧸ p`-algebra isomorphism between `B ⧸ P` and `C ⧸ Q` induced by an `A`-algebra
  isomorphism between `B` and `C`, where `Q = σ P`. -/
def algEquivOfEqMap (h : Q = P.map σ) : (B ⧸ P) ≃ₐ[A ⧸ p] (C ⧸ Q) where
  __ := quotientEquiv P Q σ h
  commutes' := by
    /-
      R✝ : Type u_1
      inst✝¹⁶ : CommRing R✝
      R : Type u_2
      inst✝¹⁵ : CommSemiring R
      A : Type u_3
      B : Type u_4
      C : Type u_5
      inst✝¹⁴ : CommRing A
      inst✝¹³ : CommRing B
      inst✝¹² : CommRing C
      inst✝¹¹ : Algebra A B
      inst✝¹⁰ : Algebra A C
      inst✝⁹ : Algebra R A
      inst✝⁸ : Algebra R B
      inst✝⁷ : IsScalarTower R A B
      P : Ideal B
      Q : Ideal C
      p : Ideal A
      inst✝⁶ : Q.LiesOver p
      inst✝⁵ : P.LiesOver p
      G : Type u_6
      inst✝⁴ : Group G
      inst✝³ : MulSemiringAction G B
      inst✝² : SMulCommClass G A B
      E : Type u_7
      inst✝¹ : EquivLike E B C
      inst✝ : AlgEquivClass E A B C
      σ : E
      h : Eq Q (Ideal.map σ P)
      ⊢ ∀ (r : HasQuotient.Quotient A p), Eq (__spread✝⁻⁰.toFun ((algebraMap (HasQuo …
    -/
    rintro ⟨x⟩
    /-
      case mk
      R✝ : Type u_1
      inst✝¹⁶ : CommRing R✝
      R : Type u_2
      inst✝¹⁵ : CommSemiring R
      A : Type u_3
      B : Type u_4
      C : Type u_5
      inst✝¹⁴ : CommRing A
      inst✝¹³ : CommRing B
      inst✝¹² : CommRing C
      inst✝¹¹ : Algebra A B
      inst✝¹⁰ : Algebra A C
      inst✝⁹ : Algebra R A
      inst✝⁸ : Algebra R B
      inst✝⁷ : IsScalarTower R A B
      P : Ideal B
      Q : Ideal C
      p : Ideal A
      inst✝⁶ : Q.LiesOver p
      inst✝⁵ : P.LiesOver p
      G : Type u_6
      inst✝⁴ : Group G
      inst✝³ : MulSemiringAction G B
      inst✝² : SMulCommClass G A B
      E : Type u_7
      inst✝¹ : EquivLike E B C
      inst✝ : AlgEquivClass E A B C
      σ : E
      h : Eq Q (Ideal.map σ P)
      r✝ : HasQuotient.Quotient A p
      x : A
      ⊢ Eq (__spread✝⁻⁰.toFun ((algebraMap (HasQuotient.Quotient A p) (HasQuotient.Q …
    -/
    exact congrArg (Ideal.Quotient.mk Q) (AlgHomClass.commutes σ x)
    /-
      🎉 no goals
    -/


@[simp]
theorem algEquivOfEqMap_apply (h : Q = P.map σ) (x : B) : algEquivOfEqMap p σ h x = σ x :=
  rfl


/-- An `A ⧸ p`-algebra isomorphism between `B ⧸ P` and `C ⧸ Q` induced by an `A`-algebra
  isomorphism between `B` and `C`, where `P = σ⁻¹ Q`. -/
def algEquivOfEqComap (h : P = Q.comap σ) : (B ⧸ P) ≃ₐ[A ⧸ p] (C ⧸ Q) :=
  algEquivOfEqMap p σ ((congrArg (map σ) h).trans (Q.map_comap_eq_self_of_equiv σ)).symm


@[simp]
theorem algEquivOfEqComap_apply (h : P = Q.comap σ) (x : B) : algEquivOfEqComap p σ h x = σ x :=
  rfl


/-- If `P` lies over `p`, then the stabilizer of `P` acts on the extension `(B ⧸ P) / (A ⧸ p)`. -/
def stabilizerHom : MulAction.stabilizer G P →* ((B ⧸ P) ≃ₐ[A ⧸ p] (B ⧸ P)) where
  toFun g := algEquivOfEqMap p (MulSemiringAction.toAlgEquiv A B g) g.2.symm
  map_one' := by
    /-
      R✝ : Type u_1
      inst✝¹⁴ : CommRing R✝
      R : Type u_2
      inst✝¹³ : CommSemiring R
      A : Type u_3
      B : Type u_4
      C : Type u_5
      inst✝¹² : CommRing A
      inst✝¹¹ : CommRing B
      inst✝¹⁰ : CommRing C
      inst✝⁹ : Algebra A B
      inst✝⁸ : Algebra A C
      inst✝⁷ : Algebra R A
      inst✝⁶ : Algebra R B
      inst✝⁵ : IsScalarTower R A B
      P : Ideal B
      Q : Ideal C
      p : Ideal A
      inst✝⁴ : Q.LiesOver p
      inst✝³ : P.LiesOver p
      G : Type u_6
      inst✝² : Group G
      inst✝¹ : MulSemiringAction G B
      inst✝ : SMulCommClass G A B
      ⊢ Eq ((fun g => Ideal.Quotient.algEquivOfEqMap p (MulSemiringAction.toAlgEquiv …
    -/
    ext ⟨x⟩
    /-
      case h.mk
      R✝ : Type u_1
      inst✝¹⁴ : CommRing R✝
      R : Type u_2
      inst✝¹³ : CommSemiring R
      A : Type u_3
      B : Type u_4
      C : Type u_5
      inst✝¹² : CommRing A
      inst✝¹¹ : CommRing B
      inst✝¹⁰ : CommRing C
      inst✝⁹ : Algebra A B
      inst✝⁸ : Algebra A C
      inst✝⁷ : Algebra R A
      inst✝⁶ : Algebra R B
      inst✝⁵ : IsScalarTower R A B
      P : Ideal B
      Q : Ideal C
      p : Ideal A
      inst✝⁴ : Q.LiesOver p
      inst✝³ : P.LiesOver p
      G : Type u_6
      inst✝² : Group G
      inst✝¹ : MulSemiringAction G B
      inst✝ : SMulCommClass G A B
      a✝ : HasQuotient.Quotient B P
      x : B
      ⊢ Eq (((fun g => Ideal.Quotient.algEquivOfEqMap p (MulSemiringAction.toAlgEqui …
    -/
    exact congrArg (Ideal.Quotient.mk P) (one_smul G x)
    /-
      🎉 no goals
    -/
  map_mul' g h := by
    /-
      R✝ : Type u_1
      inst✝¹⁴ : CommRing R✝
      R : Type u_2
      inst✝¹³ : CommSemiring R
      A : Type u_3
      B : Type u_4
      C : Type u_5
      inst✝¹² : CommRing A
      inst✝¹¹ : CommRing B
      inst✝¹⁰ : CommRing C
      inst✝⁹ : Algebra A B
      inst✝⁸ : Algebra A C
      inst✝⁷ : Algebra R A
      inst✝⁶ : Algebra R B
      inst✝⁵ : IsScalarTower R A B
      P : Ideal B
      Q : Ideal C
      p : Ideal A
      inst✝⁴ : Q.LiesOver p
      inst✝³ : P.LiesOver p
      G : Type u_6
      inst✝² : Group G
      inst✝¹ : MulSemiringAction G B
      inst✝ : SMulCommClass G A B
      g h : Subtype fun x => Membership.mem (MulAction.stabilizer G P) x
      ⊢ Eq ({ toFun := fun g => Ideal.Quotient.algEquivOfEqMap p (MulSemiringAction. …
    -/
    ext ⟨x⟩
    /-
      case h.mk
      R✝ : Type u_1
      inst✝¹⁴ : CommRing R✝
      R : Type u_2
      inst✝¹³ : CommSemiring R
      A : Type u_3
      B : Type u_4
      C : Type u_5
      inst✝¹² : CommRing A
      inst✝¹¹ : CommRing B
      inst✝¹⁰ : CommRing C
      inst✝⁹ : Algebra A B
      inst✝⁸ : Algebra A C
      inst✝⁷ : Algebra R A
      inst✝⁶ : Algebra R B
      inst✝⁵ : IsScalarTower R A B
      P : Ideal B
      Q : Ideal C
      p : Ideal A
      inst✝⁴ : Q.LiesOver p
      inst✝³ : P.LiesOver p
      G : Type u_6
      inst✝² : Group G
      inst✝¹ : MulSemiringAction G B
      inst✝ : SMulCommClass G A B
      g h : Subtype fun x => Membership.mem (MulAction.stabilizer G P) x
      a✝ : HasQuotient.Quotient B P
      x : B
      ⊢ Eq (({ toFun := fun g => Ideal.Quotient.algEquivOfEqMap p (MulSemiringAction …
    -/
    exact congrArg (Ideal.Quotient.mk P) (mul_smul g h x)
    /-
      🎉 no goals
    -/


@[simp] theorem stabilizerHom_apply (g : MulAction.stabilizer G P) (b : B) :
    stabilizerHom P p G g b = ↑(g • b) :=
  rfl


variable (A) in
/-- If `B` is an integral `A`-algebra, `P` is a maximal ideal of `B`, then the pull back of
  `P` is also a maximal ideal of `A`. -/
instance IsMaximal.under [P.IsMaximal] : (P.under A).IsMaximal :=
  isMaximal_comap_of_isIntegral_of_isMaximal P


theorem IsMaximal.of_liesOver_isMaximal [hpm : p.IsMaximal] [P.IsPrime] : P.IsMaximal := by
  /-
    A : Type u_2
    inst✝⁵ : CommRing A
    B : Type u_3
    inst✝⁴ : CommRing B
    inst✝³ : Algebra A B
    inst✝² : Algebra.IsIntegral A B
    P : Ideal B
    p : Ideal A
    inst✝¹ : P.LiesOver p
    hpm : p.IsMaximal
    inst✝ : P.IsPrime
    ⊢ P.IsMaximal
  -/
  rw [P.over_def p] at hpm
  /-
    A : Type u_2
    inst✝⁵ : CommRing A
    B : Type u_3
    inst✝⁴ : CommRing B
    inst✝³ : Algebra A B
    inst✝² : Algebra.IsIntegral A B
    P : Ideal B
    p : Ideal A
    inst✝¹ : P.LiesOver p
    hpm : (Ideal.under A P).IsMaximal
    inst✝ : P.IsPrime
    ⊢ P.IsMaximal
  -/
  exact isMaximal_of_isIntegral_of_isMaximal_comap P hpm
  /-
    🎉 no goals
  -/


theorem IsMaximal.of_isMaximal_liesOver [P.IsMaximal] : p.IsMaximal := by
  /-
    A : Type u_2
    inst✝⁵ : CommRing A
    B : Type u_3
    inst✝⁴ : CommRing B
    inst✝³ : Algebra A B
    inst✝² : Algebra.IsIntegral A B
    P : Ideal B
    p : Ideal A
    inst✝¹ : P.LiesOver p
    inst✝ : P.IsMaximal
    ⊢ p.IsMaximal
  -/
  rw [P.over_def p]
  /-
    A : Type u_2
    inst✝⁵ : CommRing A
    B : Type u_3
    inst✝⁴ : CommRing B
    inst✝³ : Algebra A B
    inst✝² : Algebra.IsIntegral A B
    P : Ideal B
    p : Ideal A
    inst✝¹ : P.LiesOver p
    inst✝ : P.IsMaximal
    ⊢ (Ideal.under A P).IsMaximal
  -/
  exact isMaximal_comap_of_isIntegral_of_isMaximal P
  /-
    🎉 no goals
  -/


/-- `B ⧸ P` is an integral `A ⧸ p`-algebra if `B` is a integral `A`-algebra. -/
instance Quotient.algebra_isIntegral_of_liesOver : Algebra.IsIntegral (A ⧸ p) (B ⧸ P) :=
  Algebra.IsIntegral.tower_top A


theorem exists_ideal_liesOver_maximal_of_isIntegral [p.IsMaximal] (B : Type*) [CommRing B]
    [Nontrivial B] [Algebra A B] [NoZeroSMulDivisors A B] [Algebra.IsIntegral A B] :
    ∃ P : Ideal B, P.IsMaximal ∧ P.LiesOver p := by
  rcases exists_ideal_over_maximal_of_isIntegral p <|
    (NoZeroSMulDivisors.ker_algebraMap_eq_bot A B).trans_le bot_le with ⟨P, hm, hP⟩
  /-
    case intro.intro
    A : Type u_2
    inst✝⁶ : CommRing A
    p : Ideal A
    inst✝⁵ : p.IsMaximal
    B : Type u_4
    inst✝⁴ : CommRing B
    inst✝³ : Nontrivial B
    inst✝² : Algebra A B
    inst✝¹ : NoZeroSMulDivisors A B
    inst✝ : Algebra.IsIntegral A B
    P : Ideal B
    hm : P.IsMaximal
    hP : Eq (Ideal.comap (algebraMap A B) P) p
    ⊢ Exists fun P => And P.IsMaximal (P.LiesOver p)
  -/
  exact ⟨P, hm, ⟨hP.symm⟩⟩
  /-
    🎉 no goals
  -/


/-- The set of all prime ideals in `B` that lie over an ideal `p` of `A`. -/
def primesOver : Set (Ideal B) :=
  { P : Ideal B | P.IsPrime ∧ P.LiesOver p }


instance primesOver.isPrime (Q : primesOver p B) : Q.1.IsPrime :=
  Q.2.1


instance primesOver.liesOver (Q : primesOver p B) : Q.1.LiesOver p :=
  Q.2.2


/-- If an ideal `P` of `B` is prime and lying over `p`, then it is in `primesOver p B`. -/
abbrev primesOver.mk (P : Ideal B) [hPp : P.IsPrime] [hp : P.LiesOver p] : primesOver p B :=
  ⟨P, ⟨hPp, hp⟩⟩


instance primesOver.isMaximal : Q.1.IsMaximal :=
  Ideal.IsMaximal.of_liesOver_isMaximal Q.1 p


variable (A B) in
lemma primesOver_bot [Nontrivial A] [IsDomain B] : primesOver (⊥ : Ideal A) B = {⊥} := by
  /-
    A : Type u_2
    inst✝⁶ : CommRing A
    B : Type u_3
    inst✝⁵ : CommRing B
    inst✝⁴ : Algebra A B
    inst✝³ : NoZeroSMulDivisors A B
    inst✝² : Algebra.IsIntegral A B
    inst✝¹ : Nontrivial A
    inst✝ : IsDomain B
    ⊢ Eq (primesOver Bot.bot B) (Singleton.singleton Bot.bot)
  -/
  ext p
  /-
    case h
    A : Type u_2
    inst✝⁶ : CommRing A
    B : Type u_3
    inst✝⁵ : CommRing B
    inst✝⁴ : Algebra A B
    inst✝³ : NoZeroSMulDivisors A B
    inst✝² : Algebra.IsIntegral A B
    inst✝¹ : Nontrivial A
    inst✝ : IsDomain B
    p : Ideal B
    ⊢ Iff (Membership.mem (primesOver Bot.bot B) p) (Membership.mem (Singleton.sin …
  -/
  refine ⟨fun ⟨_, ⟨h⟩⟩ ↦ p.eq_bot_of_comap_eq_bot h.symm, ?_⟩
  /-
    case h
    A : Type u_2
    inst✝⁶ : CommRing A
    B : Type u_3
    inst✝⁵ : CommRing B
    inst✝⁴ : Algebra A B
    inst✝³ : NoZeroSMulDivisors A B
    inst✝² : Algebra.IsIntegral A B
    inst✝¹ : Nontrivial A
    inst✝ : IsDomain B
    p : Ideal B
    ⊢ Membership.mem (Singleton.singleton Bot.bot) p → Membership.mem (primesOver  …
  -/
  rintro rfl
  /-
    case h
    A : Type u_2
    inst✝⁶ : CommRing A
    B : Type u_3
    inst✝⁵ : CommRing B
    inst✝⁴ : Algebra A B
    inst✝³ : NoZeroSMulDivisors A B
    inst✝² : Algebra.IsIntegral A B
    inst✝¹ : Nontrivial A
    inst✝ : IsDomain B
    ⊢ Membership.mem (primesOver Bot.bot B) Bot.bot
  -/
  exact ⟨Ideal.bot_prime, Ideal.bot_liesOver_bot A B⟩
  /-
    🎉 no goals
  -/


