theorem exists_isNilpotent_isSemisimple_of_separable_of_dvd_pow {P : K[X]} {k : ℕ}
    (sep : P.Separable) (nil : minpoly K f ∣ P ^ k) :
    ∃ᵉ (n ∈ adjoin K {f}) (s ∈ adjoin K {f}), IsNilpotent n ∧ IsSemisimple s ∧ f = n + s := by
  /-
    K : Type u_1
    V : Type u_2
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    f : Module.End K V
    P : Polynomial K
    k : Nat
    sep : P.Separable
    nil : Dvd.dvd (minpoly K f) (HPow.hPow P k)
    ⊢ Exists fun n => And (Membership.mem (Algebra.adjoin K (Singleton.singleton f …
  -/
  set ff : adjoin K {f} := ⟨f, self_mem_adjoin_singleton K f⟩
  /-
    K : Type u_1
    V : Type u_2
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    f : Module.End K V
    P : Polynomial K
    k : Nat
    sep : P.Separable
    nil : Dvd.dvd (minpoly K f) (HPow.hPow P k)
    ff : Subtype fun x => Membership.mem (Algebra.adjoin K (Singleton.singleton f) …
    ⊢ Exists fun n => And (Membership.mem (Algebra.adjoin K (Singleton.singleton f …
  -/
  set P' := derivative P
  have nil' : IsNilpotent (aeval ff P) := by
    use k
    obtain ⟨q, hq⟩ := nil
    rw [← map_pow, Subtype.ext_iff]
    simp [ff, hq]
  have sep' : IsUnit (aeval ff P') := by
    obtain ⟨a, b, h⟩ : IsCoprime (P ^ k) P' := sep.pow_left
    replace h : (aeval f b) * (aeval f P') = 1 := by
      simpa only [map_add, map_mul, map_one, minpoly.dvd_iff.mp nil, mul_zero, zero_add]
        using (aeval f).congr_arg h
    refine isUnit_of_mul_eq_one_right (aeval ff b) _ (Subtype.ext_iff.mpr ?_)
    simpa [ff, coe_aeval_mk_apply] using h
  /-
    K : Type u_1
    V : Type u_2
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    f : Module.End K V
    P : Polynomial K
    k : Nat
    sep : P.Separable
    nil : Dvd.dvd (minpoly K f) (HPow.hPow P k)
    ff : Subtype fun x => Membership.mem (Algebra.adjoin K (Singleton.singleton f) …
    P' : Polynomial K := Polynomial.derivative P
    nil' : IsNilpotent ((Polynomial.aeval ff) P)
    sep' : IsUnit ((Polynomial.aeval ff) P')
    ⊢ Exists fun n => And (Membership.mem (Algebra.adjoin K (Singleton.singleton f …
  -/
  obtain ⟨⟨s, mem⟩, ⟨⟨k, hk⟩, hss⟩, -⟩ := existsUnique_nilpotent_sub_and_aeval_eq_zero nil' sep'
  /-
    case intro.mk.intro.intro.intro
    K : Type u_1
    V : Type u_2
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    f : Module.End K V
    P : Polynomial K
    k✝ : Nat
    sep : P.Separable
    nil : Dvd.dvd (minpoly K f) (HPow.hPow P k✝)
    ff : Subtype fun x => Membership.mem (Algebra.adjoin K (Singleton.singleton f) …
    P' : Polynomial K := Polynomial.derivative P
    nil' : IsNilpotent ((Polynomial.aeval ff) P)
    sep' : IsUnit ((Polynomial.aeval ff) P')
    s : Module.End K V
    mem : Membership.mem (Algebra.adjoin K (Singleton.singleton f)) s
    hss : Eq ((Polynomial.aeval ⟨s, mem⟩) P) 0
    k : Nat
    hk : Eq (HPow.hPow (HSub.hSub ff ⟨s, mem⟩) k) 0
    ⊢ Exists fun n => And (Membership.mem (Algebra.adjoin K (Singleton.singleton f …
  -/
  refine ⟨f - s, ?_, s, mem, ⟨k, ?_⟩, ?_, (sub_add_cancel f s).symm⟩
    /-
      case intro.mk.intro.intro.intro.refine_1
      K : Type u_1
      V : Type u_2
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      f : Module.End K V
      P : Polynomial K
      k✝ : Nat
      sep : P.Separable
      nil : Dvd.dvd (minpoly K f) (HPow.hPow P k✝)
      ff : Subtype fun x => Membership.mem (Algebra.adjoin K (Singleton.singleton f) …
      P' : Polynomial K := Polynomial.derivative P
      nil' : IsNilpotent ((Polynomial.aeval ff) P)
      sep' : IsUnit ((Polynomial.aeval ff) P')
      s : Module.End K V
      mem : Membership.mem (Algebra.adjoin K (Singleton.singleton f)) s
      hss : Eq ((Polynomial.aeval ⟨s, mem⟩) P) 0
      k : Nat
      hk : Eq (HPow.hPow (HSub.hSub ff ⟨s, mem⟩) k) 0
      ⊢ Membership.mem (Algebra.adjoin K (Singleton.singleton f)) (HSub.hSub f s)
    -/
  · exact sub_mem (self_mem_adjoin_singleton K f) mem
    /-
      🎉 no goals
    -/
    /-
      case intro.mk.intro.intro.intro.refine_2
      K : Type u_1
      V : Type u_2
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      f : Module.End K V
      P : Polynomial K
      k✝ : Nat
      sep : P.Separable
      nil : Dvd.dvd (minpoly K f) (HPow.hPow P k✝)
      ff : Subtype fun x => Membership.mem (Algebra.adjoin K (Singleton.singleton f) …
      P' : Polynomial K := Polynomial.derivative P
      nil' : IsNilpotent ((Polynomial.aeval ff) P)
      sep' : IsUnit ((Polynomial.aeval ff) P')
      s : Module.End K V
      mem : Membership.mem (Algebra.adjoin K (Singleton.singleton f)) s
      hss : Eq ((Polynomial.aeval ⟨s, mem⟩) P) 0
      k : Nat
      hk : Eq (HPow.hPow (HSub.hSub ff ⟨s, mem⟩) k) 0
      ⊢ Eq (HPow.hPow (HSub.hSub f s) k) 0
    -/
  · rw [Subtype.ext_iff] at hk
    /-
      case intro.mk.intro.intro.intro.refine_2
      K : Type u_1
      V : Type u_2
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      f : Module.End K V
      P : Polynomial K
      k✝ : Nat
      sep : P.Separable
      nil : Dvd.dvd (minpoly K f) (HPow.hPow P k✝)
      ff : Subtype fun x => Membership.mem (Algebra.adjoin K (Singleton.singleton f) …
      P' : Polynomial K := Polynomial.derivative P
      nil' : IsNilpotent ((Polynomial.aeval ff) P)
      sep' : IsUnit ((Polynomial.aeval ff) P')
      s : Module.End K V
      mem : Membership.mem (Algebra.adjoin K (Singleton.singleton f)) s
      hss : Eq ((Polynomial.aeval ⟨s, mem⟩) P) 0
      k : Nat
      hk : Eq ↑(HPow.hPow (HSub.hSub ff ⟨s, mem⟩) k) ↑0
      ⊢ Eq (HPow.hPow (HSub.hSub f s) k) 0
    -/
    simpa using hk
    /-
      🎉 no goals
    -/
    /-
      case intro.mk.intro.intro.intro.refine_3
      K : Type u_1
      V : Type u_2
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      f : Module.End K V
      P : Polynomial K
      k✝ : Nat
      sep : P.Separable
      nil : Dvd.dvd (minpoly K f) (HPow.hPow P k✝)
      ff : Subtype fun x => Membership.mem (Algebra.adjoin K (Singleton.singleton f) …
      P' : Polynomial K := Polynomial.derivative P
      nil' : IsNilpotent ((Polynomial.aeval ff) P)
      sep' : IsUnit ((Polynomial.aeval ff) P')
      s : Module.End K V
      mem : Membership.mem (Algebra.adjoin K (Singleton.singleton f)) s
      hss : Eq ((Polynomial.aeval ⟨s, mem⟩) P) 0
      k : Nat
      hk : Eq (HPow.hPow (HSub.hSub ff ⟨s, mem⟩) k) 0
      ⊢ s.IsSemisimple
    -/
  · replace hss : aeval s P = 0 := by rwa [Subtype.ext_iff, coe_aeval_mk_apply] at hss
    /-
      case intro.mk.intro.intro.intro.refine_3
      K : Type u_1
      V : Type u_2
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      f : Module.End K V
      P : Polynomial K
      k✝ : Nat
      sep : P.Separable
      nil : Dvd.dvd (minpoly K f) (HPow.hPow P k✝)
      ff : Subtype fun x => Membership.mem (Algebra.adjoin K (Singleton.singleton f) …
      P' : Polynomial K := Polynomial.derivative P
      nil' : IsNilpotent ((Polynomial.aeval ff) P)
      sep' : IsUnit ((Polynomial.aeval ff) P')
      s : Module.End K V
      mem : Membership.mem (Algebra.adjoin K (Singleton.singleton f)) s
      k : Nat
      hk : Eq (HPow.hPow (HSub.hSub ff ⟨s, mem⟩) k) 0
      hss : Eq ((Polynomial.aeval s) P) 0
      ⊢ s.IsSemisimple
    -/
    exact isSemisimple_of_squarefree_aeval_eq_zero sep.squarefree hss
    /-
      🎉 no goals
    -/


/-- **Jordan-Chevalley-Dunford decomposition**: an endomorphism of a finite-dimensional vector space
over a perfect field may be written as a sum of nilpotent and semisimple endomorphisms. Moreover
these nilpotent and semisimple components are polynomial expressions in the original endomorphism.
-/
theorem exists_isNilpotent_isSemisimple [PerfectField K] :
    ∃ᵉ (n ∈ adjoin K {f}) (s ∈ adjoin K {f}), IsNilpotent n ∧ IsSemisimple s ∧ f = n + s := by
  /-
    K : Type u_1
    V : Type u_2
    inst✝⁴ : Field K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    f : Module.End K V
    inst✝¹ : FiniteDimensional K V
    inst✝ : PerfectField K
    ⊢ Exists fun n => And (Membership.mem (Algebra.adjoin K (Singleton.singleton f …
  -/
  obtain ⟨g, k, sep, -, nil⟩ := exists_squarefree_dvd_pow_of_ne_zero (minpoly.ne_zero_of_finite K f)
  /-
    case intro.intro.intro.intro
    K : Type u_1
    V : Type u_2
    inst✝⁴ : Field K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    f : Module.End K V
    inst✝¹ : FiniteDimensional K V
    inst✝ : PerfectField K
    g : Polynomial K
    k : Nat
    sep : Squarefree g
    nil : Dvd.dvd (minpoly K f) (HPow.hPow g k)
    ⊢ Exists fun n => And (Membership.mem (Algebra.adjoin K (Singleton.singleton f …
  -/
  rw [← PerfectField.separable_iff_squarefree] at sep
  /-
    case intro.intro.intro.intro
    K : Type u_1
    V : Type u_2
    inst✝⁴ : Field K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    f : Module.End K V
    inst✝¹ : FiniteDimensional K V
    inst✝ : PerfectField K
    g : Polynomial K
    k : Nat
    sep : g.Separable
    nil : Dvd.dvd (minpoly K f) (HPow.hPow g k)
    ⊢ Exists fun n => And (Membership.mem (Algebra.adjoin K (Singleton.singleton f …
  -/
  exact exists_isNilpotent_isSemisimple_of_separable_of_dvd_pow sep nil
  /-
    🎉 no goals
  -/


