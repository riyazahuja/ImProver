theorem eq_bot_of_generator_maximal_map_eq_zero (b : Basis ι R M) {N : Submodule R M}
    {ϕ : M →ₗ[R] R} (hϕ : ∀ ψ : M →ₗ[R] R, ¬N.map ϕ < N.map ψ) [(N.map ϕ).IsPrincipal]
    (hgen : generator (N.map ϕ) = (0 : R)) : N = ⊥ := by
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type u_1
    b : Basis ι R M
    N : Submodule R M
    ϕ : LinearMap (RingHom.id R) M R
    hϕ : ∀ (ψ : LinearMap (RingHom.id R) M R), Not (LT.lt (Submodule.map ϕ N) (Sub …
    inst✝ : (Submodule.map ϕ N).IsPrincipal
    hgen : Eq (Submodule.IsPrincipal.generator (Submodule.map ϕ N)) 0
    ⊢ Eq N Bot.bot
  -/
  rw [Submodule.eq_bot_iff]
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type u_1
    b : Basis ι R M
    N : Submodule R M
    ϕ : LinearMap (RingHom.id R) M R
    hϕ : ∀ (ψ : LinearMap (RingHom.id R) M R), Not (LT.lt (Submodule.map ϕ N) (Sub …
    inst✝ : (Submodule.map ϕ N).IsPrincipal
    hgen : Eq (Submodule.IsPrincipal.generator (Submodule.map ϕ N)) 0
    ⊢ ∀ (x : M), Membership.mem N x → Eq x 0
  -/
  intro x hx
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type u_1
    b : Basis ι R M
    N : Submodule R M
    ϕ : LinearMap (RingHom.id R) M R
    hϕ : ∀ (ψ : LinearMap (RingHom.id R) M R), Not (LT.lt (Submodule.map ϕ N) (Sub …
    inst✝ : (Submodule.map ϕ N).IsPrincipal
    hgen : Eq (Submodule.IsPrincipal.generator (Submodule.map ϕ N)) 0
    x : M
    hx : Membership.mem N x
    ⊢ Eq x 0
  -/
  refine b.ext_elem fun i ↦ ?_
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type u_1
    b : Basis ι R M
    N : Submodule R M
    ϕ : LinearMap (RingHom.id R) M R
    hϕ : ∀ (ψ : LinearMap (RingHom.id R) M R), Not (LT.lt (Submodule.map ϕ N) (Sub …
    inst✝ : (Submodule.map ϕ N).IsPrincipal
    hgen : Eq (Submodule.IsPrincipal.generator (Submodule.map ϕ N)) 0
    x : M
    hx : Membership.mem N x
    i : ι
    ⊢ Eq ((b.repr x) i) ((b.repr 0) i)
  -/
  rw [(eq_bot_iff_generator_eq_zero _).mpr hgen] at hϕ
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type u_1
    b : Basis ι R M
    N : Submodule R M
    ϕ : LinearMap (RingHom.id R) M R
    hϕ : ∀ (ψ : LinearMap (RingHom.id R) M R), Not (LT.lt Bot.bot (Submodule.map ψ …
    inst✝ : (Submodule.map ϕ N).IsPrincipal
    hgen : Eq (Submodule.IsPrincipal.generator (Submodule.map ϕ N)) 0
    x : M
    hx : Membership.mem N x
    i : ι
    ⊢ Eq ((b.repr x) i) ((b.repr 0) i)
  -/
  rw [LinearEquiv.map_zero, Finsupp.zero_apply]
  exact
    (Submodule.eq_bot_iff _).mp (not_bot_lt_iff.1 <| hϕ (Finsupp.lapply i ∘ₗ ↑b.repr)) _
      ⟨x, hx, rfl⟩


theorem eq_bot_of_generator_maximal_submoduleImage_eq_zero {N O : Submodule R M} (b : Basis ι R O)
    (hNO : N ≤ O) {ϕ : O →ₗ[R] R} (hϕ : ∀ ψ : O →ₗ[R] R, ¬ϕ.submoduleImage N < ψ.submoduleImage N)
    [(ϕ.submoduleImage N).IsPrincipal] (hgen : generator (ϕ.submoduleImage N) = 0) : N = ⊥ := by
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type u_1
    N O : Submodule R M
    b : Basis ι R (Subtype fun x => Membership.mem O x)
    hNO : LE.le N O
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R
    hϕ : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R), …
    inst✝ : (ϕ.submoduleImage N).IsPrincipal
    hgen : Eq (Submodule.IsPrincipal.generator (ϕ.submoduleImage N)) 0
    ⊢ Eq N Bot.bot
  -/
  rw [Submodule.eq_bot_iff]
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type u_1
    N O : Submodule R M
    b : Basis ι R (Subtype fun x => Membership.mem O x)
    hNO : LE.le N O
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R
    hϕ : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R), …
    inst✝ : (ϕ.submoduleImage N).IsPrincipal
    hgen : Eq (Submodule.IsPrincipal.generator (ϕ.submoduleImage N)) 0
    ⊢ ∀ (x : M), Membership.mem N x → Eq x 0
  -/
  intro x hx
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type u_1
    N O : Submodule R M
    b : Basis ι R (Subtype fun x => Membership.mem O x)
    hNO : LE.le N O
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R
    hϕ : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R), …
    inst✝ : (ϕ.submoduleImage N).IsPrincipal
    hgen : Eq (Submodule.IsPrincipal.generator (ϕ.submoduleImage N)) 0
    x : M
    hx : Membership.mem N x
    ⊢ Eq x 0
  -/
  refine (mk_eq_zero _ _).mp (show (⟨x, hNO hx⟩ : O) = 0 from b.ext_elem fun i ↦ ?_)
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type u_1
    N O : Submodule R M
    b : Basis ι R (Subtype fun x => Membership.mem O x)
    hNO : LE.le N O
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R
    hϕ : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R), …
    inst✝ : (ϕ.submoduleImage N).IsPrincipal
    hgen : Eq (Submodule.IsPrincipal.generator (ϕ.submoduleImage N)) 0
    x : M
    hx : Membership.mem N x
    i : ι
    ⊢ Eq ((b.repr ⟨x, ⋯⟩) i) ((b.repr 0) i)
  -/
  rw [(eq_bot_iff_generator_eq_zero _).mpr hgen] at hϕ
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type u_1
    N O : Submodule R M
    b : Basis ι R (Subtype fun x => Membership.mem O x)
    hNO : LE.le N O
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R
    hϕ : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R), …
    inst✝ : (ϕ.submoduleImage N).IsPrincipal
    hgen : Eq (Submodule.IsPrincipal.generator (ϕ.submoduleImage N)) 0
    x : M
    hx : Membership.mem N x
    i : ι
    ⊢ Eq ((b.repr ⟨x, ⋯⟩) i) ((b.repr 0) i)
  -/
  rw [LinearEquiv.map_zero, Finsupp.zero_apply]
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type u_1
    N O : Submodule R M
    b : Basis ι R (Subtype fun x => Membership.mem O x)
    hNO : LE.le N O
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R
    hϕ : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R), …
    inst✝ : (ϕ.submoduleImage N).IsPrincipal
    hgen : Eq (Submodule.IsPrincipal.generator (ϕ.submoduleImage N)) 0
    x : M
    hx : Membership.mem N x
    i : ι
    ⊢ Eq ((b.repr ⟨x, ⋯⟩) i) 0
  -/
  refine (Submodule.eq_bot_iff _).mp (not_bot_lt_iff.1 <| hϕ (Finsupp.lapply i ∘ₗ ↑b.repr)) _ ?_
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type u_1
    N O : Submodule R M
    b : Basis ι R (Subtype fun x => Membership.mem O x)
    hNO : LE.le N O
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R
    hϕ : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R), …
    inst✝ : (ϕ.submoduleImage N).IsPrincipal
    hgen : Eq (Submodule.IsPrincipal.generator (ϕ.submoduleImage N)) 0
    x : M
    hx : Membership.mem N x
    i : ι
    ⊢ Membership.mem (((Finsupp.lapply i).comp ↑b.repr).submoduleImage N) ((b.repr …
  -/
  exact (LinearMap.mem_submoduleImage_of_le hNO).mpr ⟨x, hx, rfl⟩
  /-
    🎉 no goals
  -/


theorem dvd_generator_iff {I : Ideal R} [I.IsPrincipal] {x : R} (hx : x ∈ I) :
    x ∣ generator I ↔ I = Ideal.span {x} := by
  /-
    R : Type u_2
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    I : Ideal R
    inst✝ : Submodule.IsPrincipal I
    x : R
    hx : Membership.mem I x
    ⊢ Iff (Dvd.dvd x (Submodule.IsPrincipal.generator I)) (Eq I (Ideal.span (Singl …
  -/
  conv_rhs => rw [← span_singleton_generator I]
  rw [Ideal.submodule_span_eq, Ideal.span_singleton_eq_span_singleton, ← dvd_dvd_iff_associated,
    ← mem_iff_generator_dvd]
  /-
    R : Type u_2
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    I : Ideal R
    inst✝ : Submodule.IsPrincipal I
    x : R
    hx : Membership.mem I x
    ⊢ Iff (Dvd.dvd x (Submodule.IsPrincipal.generator I)) (And (Membership.mem I x …
  -/
  exact ⟨fun h ↦ ⟨hx, h⟩, fun h ↦ h.2⟩
  /-
    🎉 no goals
  -/


theorem generator_maximal_submoduleImage_dvd {N O : Submodule R M} (hNO : N ≤ O) {ϕ : O →ₗ[R] R}
    (hϕ : ∀ ψ : O →ₗ[R] R, ¬ϕ.submoduleImage N < ψ.submoduleImage N)
    [(ϕ.submoduleImage N).IsPrincipal] (y : M) (yN : y ∈ N)
    (ϕy_eq : ϕ ⟨y, hNO yN⟩ = generator (ϕ.submoduleImage N)) (ψ : O →ₗ[R] R) :
    generator (ϕ.submoduleImage N) ∣ ψ ⟨y, hNO yN⟩ := by
  /-
    R : Type u_2
    inst✝⁵ : CommRing R
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : IsDomain R
    inst✝¹ : IsPrincipalIdealRing R
    N O : Submodule R M
    hNO : LE.le N O
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R
    hϕ : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R), …
    inst✝ : (ϕ.submoduleImage N).IsPrincipal
    y : M
    yN : Membership.mem N y
    ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) (Submodule.IsPrincipal.generator (ϕ.submoduleImage N))
    ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R
    ⊢ Dvd.dvd (Submodule.IsPrincipal.generator (ϕ.submoduleImage N)) (ψ ⟨y, ⋯⟩)
  -/
  let a : R := generator (ϕ.submoduleImage N)
  /-
    R : Type u_2
    inst✝⁵ : CommRing R
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : IsDomain R
    inst✝¹ : IsPrincipalIdealRing R
    N O : Submodule R M
    hNO : LE.le N O
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R
    hϕ : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R), …
    inst✝ : (ϕ.submoduleImage N).IsPrincipal
    y : M
    yN : Membership.mem N y
    ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) (Submodule.IsPrincipal.generator (ϕ.submoduleImage N))
    ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R
    a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
    ⊢ Dvd.dvd (Submodule.IsPrincipal.generator (ϕ.submoduleImage N)) (ψ ⟨y, ⋯⟩)
  -/
  let d : R := IsPrincipal.generator (Submodule.span R {a, ψ ⟨y, hNO yN⟩})
  /-
    R : Type u_2
    inst✝⁵ : CommRing R
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : IsDomain R
    inst✝¹ : IsPrincipalIdealRing R
    N O : Submodule R M
    hNO : LE.le N O
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R
    hϕ : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R), …
    inst✝ : (ϕ.submoduleImage N).IsPrincipal
    y : M
    yN : Membership.mem N y
    ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) (Submodule.IsPrincipal.generator (ϕ.submoduleImage N))
    ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R
    a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
    d : R := Submodule.IsPrincipal.generator (Submodule.span R (Insert.insert a (S …
    ⊢ Dvd.dvd (Submodule.IsPrincipal.generator (ϕ.submoduleImage N)) (ψ ⟨y, ⋯⟩)
  -/
  have d_dvd_left : d ∣ a := (mem_iff_generator_dvd _).mp (subset_span (mem_insert _ _))
  have d_dvd_right : d ∣ ψ ⟨y, hNO yN⟩ :=
    (mem_iff_generator_dvd _).mp (subset_span (mem_insert_of_mem _ (mem_singleton _)))
  /-
    R : Type u_2
    inst✝⁵ : CommRing R
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : IsDomain R
    inst✝¹ : IsPrincipalIdealRing R
    N O : Submodule R M
    hNO : LE.le N O
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R
    hϕ : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R), …
    inst✝ : (ϕ.submoduleImage N).IsPrincipal
    y : M
    yN : Membership.mem N y
    ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) (Submodule.IsPrincipal.generator (ϕ.submoduleImage N))
    ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R
    a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
    d : R := Submodule.IsPrincipal.generator (Submodule.span R (Insert.insert a (S …
    d_dvd_left : Dvd.dvd d a
    d_dvd_right : Dvd.dvd d (ψ ⟨y, ⋯⟩)
    ⊢ Dvd.dvd (Submodule.IsPrincipal.generator (ϕ.submoduleImage N)) (ψ ⟨y, ⋯⟩)
  -/
  refine dvd_trans ?_ d_dvd_right
  rw [dvd_generator_iff, Ideal.span, ←
    span_singleton_generator (Submodule.span R {a, ψ ⟨y, hNO yN⟩})]
  · obtain ⟨r₁, r₂, d_eq⟩ : ∃ r₁ r₂ : R, d = r₁ * a + r₂ * ψ ⟨y, hNO yN⟩ := by
      obtain ⟨r₁, r₂', hr₂', hr₁⟩ :=
        mem_span_insert.mp (IsPrincipal.generator_mem (Submodule.span R {a, ψ ⟨y, hNO yN⟩}))
      obtain ⟨r₂, rfl⟩ := mem_span_singleton.mp hr₂'
      exact ⟨r₁, r₂, hr₁⟩
    /-
      case intro.intro
      R : Type u_2
      inst✝⁵ : CommRing R
      M : Type u_3
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : IsDomain R
      inst✝¹ : IsPrincipalIdealRing R
      N O : Submodule R M
      hNO : LE.le N O
      ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R
      hϕ : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R), …
      inst✝ : (ϕ.submoduleImage N).IsPrincipal
      y : M
      yN : Membership.mem N y
      ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) (Submodule.IsPrincipal.generator (ϕ.submoduleImage N))
      ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R
      a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
      d : R := Submodule.IsPrincipal.generator (Submodule.span R (Insert.insert a (S …
      d_dvd_left : Dvd.dvd d a
      d_dvd_right : Dvd.dvd d (ψ ⟨y, ⋯⟩)
      r₁ r₂ : R
      d_eq : Eq d (HAdd.hAdd (HMul.hMul r₁ a) (HMul.hMul r₂ (ψ ⟨y, ⋯⟩)))
      ⊢ Eq (Submodule.span R (Singleton.singleton (Submodule.IsPrincipal.generator ( …
    -/
    let ψ' : O →ₗ[R] R := r₁ • ϕ + r₂ • ψ
    have : span R {d} ≤ ψ'.submoduleImage N := by
      rw [span_le, singleton_subset_iff, SetLike.mem_coe, LinearMap.mem_submoduleImage_of_le hNO]
      refine ⟨y, yN, ?_⟩
      change r₁ * ϕ ⟨y, hNO yN⟩ + r₂ * ψ ⟨y, hNO yN⟩ = d
      rw [d_eq, ϕy_eq]
    refine
      le_antisymm (this.trans (le_of_eq ?_)) (Ideal.span_singleton_le_span_singleton.mpr d_dvd_left)
    /-
      case intro.intro
      R : Type u_2
      inst✝⁵ : CommRing R
      M : Type u_3
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : IsDomain R
      inst✝¹ : IsPrincipalIdealRing R
      N O : Submodule R M
      hNO : LE.le N O
      ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R
      hϕ : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R), …
      inst✝ : (ϕ.submoduleImage N).IsPrincipal
      y : M
      yN : Membership.mem N y
      ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) (Submodule.IsPrincipal.generator (ϕ.submoduleImage N))
      ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R
      a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
      d : R := Submodule.IsPrincipal.generator (Submodule.span R (Insert.insert a (S …
      d_dvd_left : Dvd.dvd d a
      d_dvd_right : Dvd.dvd d (ψ ⟨y, ⋯⟩)
      r₁ r₂ : R
      d_eq : Eq d (HAdd.hAdd (HMul.hMul r₁ a) (HMul.hMul r₂ (ψ ⟨y, ⋯⟩)))
      ψ' : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R := HAdd. …
      this : LE.le (Submodule.span R (Singleton.singleton d)) (ψ'.submoduleImage N)
      ⊢ Eq (ψ'.submoduleImage N) (Submodule.span R (Singleton.singleton (Submodule.I …
    -/
    rw [span_singleton_generator]
    /-
      case intro.intro
      R : Type u_2
      inst✝⁵ : CommRing R
      M : Type u_3
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : IsDomain R
      inst✝¹ : IsPrincipalIdealRing R
      N O : Submodule R M
      hNO : LE.le N O
      ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R
      hϕ : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R), …
      inst✝ : (ϕ.submoduleImage N).IsPrincipal
      y : M
      yN : Membership.mem N y
      ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) (Submodule.IsPrincipal.generator (ϕ.submoduleImage N))
      ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R
      a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
      d : R := Submodule.IsPrincipal.generator (Submodule.span R (Insert.insert a (S …
      d_dvd_left : Dvd.dvd d a
      d_dvd_right : Dvd.dvd d (ψ ⟨y, ⋯⟩)
      r₁ r₂ : R
      d_eq : Eq d (HAdd.hAdd (HMul.hMul r₁ a) (HMul.hMul r₂ (ψ ⟨y, ⋯⟩)))
      ψ' : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R := HAdd. …
      this : LE.le (Submodule.span R (Singleton.singleton d)) (ψ'.submoduleImage N)
      ⊢ Eq (ψ'.submoduleImage N) (ϕ.submoduleImage N)
    -/
    apply (le_trans _ this).eq_of_not_gt (hϕ ψ')
    /-
      R : Type u_2
      inst✝⁵ : CommRing R
      M : Type u_3
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : IsDomain R
      inst✝¹ : IsPrincipalIdealRing R
      N O : Submodule R M
      hNO : LE.le N O
      ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R
      hϕ : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R), …
      inst✝ : (ϕ.submoduleImage N).IsPrincipal
      y : M
      yN : Membership.mem N y
      ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) (Submodule.IsPrincipal.generator (ϕ.submoduleImage N))
      ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R
      a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
      d : R := Submodule.IsPrincipal.generator (Submodule.span R (Insert.insert a (S …
      d_dvd_left : Dvd.dvd d a
      d_dvd_right : Dvd.dvd d (ψ ⟨y, ⋯⟩)
      r₁ r₂ : R
      d_eq : Eq d (HAdd.hAdd (HMul.hMul r₁ a) (HMul.hMul r₂ (ψ ⟨y, ⋯⟩)))
      ψ' : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R := HAdd. …
      this : LE.le (Submodule.span R (Singleton.singleton d)) (ψ'.submoduleImage N)
      ⊢ LE.le (ϕ.submoduleImage N) (Submodule.span R (Singleton.singleton d))
    -/
    rw [← span_singleton_generator (ϕ.submoduleImage N)]
    /-
      R : Type u_2
      inst✝⁵ : CommRing R
      M : Type u_3
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : IsDomain R
      inst✝¹ : IsPrincipalIdealRing R
      N O : Submodule R M
      hNO : LE.le N O
      ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R
      hϕ : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R), …
      inst✝ : (ϕ.submoduleImage N).IsPrincipal
      y : M
      yN : Membership.mem N y
      ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) (Submodule.IsPrincipal.generator (ϕ.submoduleImage N))
      ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R
      a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
      d : R := Submodule.IsPrincipal.generator (Submodule.span R (Insert.insert a (S …
      d_dvd_left : Dvd.dvd d a
      d_dvd_right : Dvd.dvd d (ψ ⟨y, ⋯⟩)
      r₁ r₂ : R
      d_eq : Eq d (HAdd.hAdd (HMul.hMul r₁ a) (HMul.hMul r₂ (ψ ⟨y, ⋯⟩)))
      ψ' : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R := HAdd. …
      this : LE.le (Submodule.span R (Singleton.singleton d)) (ψ'.submoduleImage N)
      ⊢ LE.le (Submodule.span R (Singleton.singleton (Submodule.IsPrincipal.generato …
    -/
    exact Ideal.span_singleton_le_span_singleton.mpr d_dvd_left
    /-
      🎉 no goals
    -/
    /-
      R : Type u_2
      inst✝⁵ : CommRing R
      M : Type u_3
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : IsDomain R
      inst✝¹ : IsPrincipalIdealRing R
      N O : Submodule R M
      hNO : LE.le N O
      ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R
      hϕ : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R), …
      inst✝ : (ϕ.submoduleImage N).IsPrincipal
      y : M
      yN : Membership.mem N y
      ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) (Submodule.IsPrincipal.generator (ϕ.submoduleImage N))
      ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R
      a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
      d : R := Submodule.IsPrincipal.generator (Submodule.span R (Insert.insert a (S …
      d_dvd_left : Dvd.dvd d a
      d_dvd_right : Dvd.dvd d (ψ ⟨y, ⋯⟩)
      ⊢ Membership.mem (Submodule.span R (Insert.insert a (Singleton.singleton (ψ ⟨y …
    -/
  · exact subset_span (mem_insert _ _)
    /-
      🎉 no goals
    -/


/-- The induction hypothesis of `Submodule.basisOfPid` and `Submodule.smithNormalForm`.

Basically, it says: let `N ≤ M` be a pair of submodules, then we can find a pair of
submodules `N' ≤ M'` of strictly smaller rank, whose basis we can extend to get a basis
of `N` and `M`. Moreover, if the basis for `M'` is up to scalars a basis for `N'`,
then the basis we find for `M` is up to scalars a basis for `N`.

For `basis_of_pid` we only need the first half and can fix `M = ⊤`,
for `smith_normal_form` we need the full statement,
but must also feed in a basis for `M` using `basis_of_pid` to keep the induction going.
-/
theorem Submodule.basis_of_pid_aux [Finite ι] {O : Type*} [AddCommGroup O] [Module R O]
    (M N : Submodule R O) (b'M : Basis ι R M) (N_bot : N ≠ ⊥) (N_le_M : N ≤ M) :
    ∃ y ∈ M, ∃ a : R, a • y ∈ N ∧ ∃ M' ≤ M, ∃ N' ≤ N,
      N' ≤ M' ∧ (∀ (c : R) (z : O), z ∈ M' → c • y + z = 0 → c = 0) ∧
      (∀ (c : R) (z : O), z ∈ N' → c • a • y + z = 0 → c = 0) ∧
      ∀ (n') (bN' : Basis (Fin n') R N'),
        ∃ bN : Basis (Fin (n' + 1)) R N,
          ∀ (m') (hn'm' : n' ≤ m') (bM' : Basis (Fin m') R M'),
            ∃ (hnm : n' + 1 ≤ m' + 1) (bM : Basis (Fin (m' + 1)) R M),
              ∀ as : Fin n' → R,
                (∀ i : Fin n', (bN' i : O) = as i • (bM' (Fin.castLE hn'm' i) : O)) →
                  ∃ as' : Fin (n' + 1) → R,
                    ∀ i : Fin (n' + 1), (bN i : O) = as' i • (bM (Fin.castLE hnm i) : O) := by
  -- Let `ϕ` be a maximal projection of `M` onto `R`, in the sense that there is
  -- no `ψ` whose image of `N` is larger than `ϕ`'s image of `N`.
  have : ∃ ϕ : M →ₗ[R] R, ∀ ψ : M →ₗ[R] R, ¬ϕ.submoduleImage N < ψ.submoduleImage N := by
    obtain ⟨P, P_eq, P_max⟩ :=
      set_has_maximal_iff_noetherian.mpr (inferInstance : IsNoetherian R R) _
        (show (Set.range fun ψ : M →ₗ[R] R ↦ ψ.submoduleImage N).Nonempty from
          ⟨_, Set.mem_range.mpr ⟨0, rfl⟩⟩)
    obtain ⟨ϕ, rfl⟩ := Set.mem_range.mp P_eq
    exact ⟨ϕ, fun ψ hψ ↦ P_max _ ⟨_, rfl⟩ hψ⟩
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : Finite ι
    O : Type u_4
    inst✝¹ : AddCommGroup O
    inst✝ : Module R O
    M N : Submodule R O
    b'M : Basis ι R (Subtype fun x => Membership.mem M x)
    N_bot : Ne N Bot.bot
    N_le_M : LE.le N M
    this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
    ⊢ Exists fun y => And (Membership.mem M y) (Exists fun a => And (Membership.me …
  -/
  let ϕ := this.choose
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : Finite ι
    O : Type u_4
    inst✝¹ : AddCommGroup O
    inst✝ : Module R O
    M N : Submodule R O
    b'M : Basis ι R (Subtype fun x => Membership.mem M x)
    N_bot : Ne N Bot.bot
    N_le_M : LE.le N M
    this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
    ⊢ Exists fun y => And (Membership.mem M y) (Exists fun a => And (Membership.me …
  -/
  have ϕ_max := this.choose_spec
  -- Since `ϕ(N)` is an `R`-submodule of the PID `R`,
  -- it is principal and generated by some `a`.
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : Finite ι
    O : Type u_4
    inst✝¹ : AddCommGroup O
    inst✝ : Module R O
    M N : Submodule R O
    b'M : Basis ι R (Subtype fun x => Membership.mem M x)
    N_bot : Ne N Bot.bot
    N_le_M : LE.le N M
    this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
    ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
    ⊢ Exists fun y => And (Membership.mem M y) (Exists fun a => And (Membership.me …
  -/
  let a := generator (ϕ.submoduleImage N)
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : Finite ι
    O : Type u_4
    inst✝¹ : AddCommGroup O
    inst✝ : Module R O
    M N : Submodule R O
    b'M : Basis ι R (Subtype fun x => Membership.mem M x)
    N_bot : Ne N Bot.bot
    N_le_M : LE.le N M
    this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
    ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
    a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
    ⊢ Exists fun y => And (Membership.mem M y) (Exists fun a => And (Membership.me …
  -/
  have a_mem : a ∈ ϕ.submoduleImage N := generator_mem _
  -- If `a` is zero, then the submodule is trivial. So let's assume `a ≠ 0`, `N ≠ ⊥`.
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : Finite ι
    O : Type u_4
    inst✝¹ : AddCommGroup O
    inst✝ : Module R O
    M N : Submodule R O
    b'M : Basis ι R (Subtype fun x => Membership.mem M x)
    N_bot : Ne N Bot.bot
    N_le_M : LE.le N M
    this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
    ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
    a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
    a_mem : Membership.mem (ϕ.submoduleImage N) a
    ⊢ Exists fun y => And (Membership.mem M y) (Exists fun a => And (Membership.me …
  -/
  by_cases a_zero : a = 0
    /-
      case pos
      ι : Type u_1
      R : Type u_2
      inst✝⁵ : CommRing R
      inst✝⁴ : IsDomain R
      inst✝³ : IsPrincipalIdealRing R
      inst✝² : Finite ι
      O : Type u_4
      inst✝¹ : AddCommGroup O
      inst✝ : Module R O
      M N : Submodule R O
      b'M : Basis ι R (Subtype fun x => Membership.mem M x)
      N_bot : Ne N Bot.bot
      N_le_M : LE.le N M
      this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
      ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
      ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
      a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
      a_mem : Membership.mem (ϕ.submoduleImage N) a
      a_zero : Eq a 0
      ⊢ Exists fun y => And (Membership.mem M y) (Exists fun a => And (Membership.me …
    -/
  · have := eq_bot_of_generator_maximal_submoduleImage_eq_zero b'M N_le_M ϕ_max a_zero
    /-
      case pos
      ι : Type u_1
      R : Type u_2
      inst✝⁵ : CommRing R
      inst✝⁴ : IsDomain R
      inst✝³ : IsPrincipalIdealRing R
      inst✝² : Finite ι
      O : Type u_4
      inst✝¹ : AddCommGroup O
      inst✝ : Module R O
      M N : Submodule R O
      b'M : Basis ι R (Subtype fun x => Membership.mem M x)
      N_bot : Ne N Bot.bot
      N_le_M : LE.le N M
      this✝ : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Memb …
      ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this✝. …
      ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
      a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
      a_mem : Membership.mem (ϕ.submoduleImage N) a
      a_zero : Eq a 0
      this : Eq N Bot.bot
      ⊢ Exists fun y => And (Membership.mem M y) (Exists fun a => And (Membership.me …
    -/
    contradiction
    /-
      🎉 no goals
    -/
  -- We claim that `ϕ⁻¹ a = y` can be taken as basis element of `N`.
  /-
    case neg
    ι : Type u_1
    R : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : Finite ι
    O : Type u_4
    inst✝¹ : AddCommGroup O
    inst✝ : Module R O
    M N : Submodule R O
    b'M : Basis ι R (Subtype fun x => Membership.mem M x)
    N_bot : Ne N Bot.bot
    N_le_M : LE.le N M
    this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
    ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
    a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
    a_mem : Membership.mem (ϕ.submoduleImage N) a
    a_zero : Not (Eq a 0)
    ⊢ Exists fun y => And (Membership.mem M y) (Exists fun a => And (Membership.me …
  -/
  obtain ⟨y, yN, ϕy_eq⟩ := (LinearMap.mem_submoduleImage_of_le N_le_M).mp a_mem
  /-
    case neg.intro.intro
    ι : Type u_1
    R : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : Finite ι
    O : Type u_4
    inst✝¹ : AddCommGroup O
    inst✝ : Module R O
    M N : Submodule R O
    b'M : Basis ι R (Subtype fun x => Membership.mem M x)
    N_bot : Ne N Bot.bot
    N_le_M : LE.le N M
    this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
    ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
    a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
    a_mem : Membership.mem (ϕ.submoduleImage N) a
    a_zero : Not (Eq a 0)
    y : O
    yN : Membership.mem N y
    ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
    ⊢ Exists fun y => And (Membership.mem M y) (Exists fun a => And (Membership.me …
  -/
  have _ϕy_ne_zero : ϕ ⟨y, N_le_M yN⟩ ≠ 0 := fun h ↦ a_zero (ϕy_eq.symm.trans h)
  -- Write `y` as `a • y'` for some `y'`.
  have hdvd : ∀ i, a ∣ b'M.coord i ⟨y, N_le_M yN⟩ := fun i ↦
    generator_maximal_submoduleImage_dvd N_le_M ϕ_max y yN ϕy_eq (b'M.coord i)
  /-
    case neg.intro.intro
    ι : Type u_1
    R : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : Finite ι
    O : Type u_4
    inst✝¹ : AddCommGroup O
    inst✝ : Module R O
    M N : Submodule R O
    b'M : Basis ι R (Subtype fun x => Membership.mem M x)
    N_bot : Ne N Bot.bot
    N_le_M : LE.le N M
    this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
    ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
    a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
    a_mem : Membership.mem (ϕ.submoduleImage N) a
    a_zero : Not (Eq a 0)
    y : O
    yN : Membership.mem N y
    ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
    _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
    hdvd : ∀ (i : ι), Dvd.dvd a ((b'M.coord i) ⟨y, ⋯⟩)
    ⊢ Exists fun y => And (Membership.mem M y) (Exists fun a => And (Membership.me …
  -/
  choose c hc using hdvd
  /-
    case neg.intro.intro
    ι : Type u_1
    R : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : Finite ι
    O : Type u_4
    inst✝¹ : AddCommGroup O
    inst✝ : Module R O
    M N : Submodule R O
    b'M : Basis ι R (Subtype fun x => Membership.mem M x)
    N_bot : Ne N Bot.bot
    N_le_M : LE.le N M
    this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
    ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
    a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
    a_mem : Membership.mem (ϕ.submoduleImage N) a
    a_zero : Not (Eq a 0)
    y : O
    yN : Membership.mem N y
    ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
    _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
    c : ι → R
    hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
    ⊢ Exists fun y => And (Membership.mem M y) (Exists fun a => And (Membership.me …
  -/
  cases nonempty_fintype ι
  /-
    case neg.intro.intro.intro
    ι : Type u_1
    R : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : Finite ι
    O : Type u_4
    inst✝¹ : AddCommGroup O
    inst✝ : Module R O
    M N : Submodule R O
    b'M : Basis ι R (Subtype fun x => Membership.mem M x)
    N_bot : Ne N Bot.bot
    N_le_M : LE.le N M
    this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
    ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
    a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
    a_mem : Membership.mem (ϕ.submoduleImage N) a
    a_zero : Not (Eq a 0)
    y : O
    yN : Membership.mem N y
    ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
    _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
    c : ι → R
    hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
    val✝ : Fintype ι
    ⊢ Exists fun y => And (Membership.mem M y) (Exists fun a => And (Membership.me …
  -/
  let y' : O := ∑ i, c i • b'M i
  /-
    case neg.intro.intro.intro
    ι : Type u_1
    R : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : Finite ι
    O : Type u_4
    inst✝¹ : AddCommGroup O
    inst✝ : Module R O
    M N : Submodule R O
    b'M : Basis ι R (Subtype fun x => Membership.mem M x)
    N_bot : Ne N Bot.bot
    N_le_M : LE.le N M
    this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
    ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
    a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
    a_mem : Membership.mem (ϕ.submoduleImage N) a
    a_zero : Not (Eq a 0)
    y : O
    yN : Membership.mem N y
    ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
    _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
    c : ι → R
    hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
    val✝ : Fintype ι
    y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
    ⊢ Exists fun y => And (Membership.mem M y) (Exists fun a => And (Membership.me …
  -/
  have y'M : y' ∈ M := M.sum_mem fun i _ ↦ M.smul_mem (c i) (b'M i).2
  have mk_y' : (⟨y', y'M⟩ : M) = ∑ i, c i • b'M i :=
    Subtype.ext
      (show y' = M.subtype _ by
        simp only [map_sum, map_smul]
        rfl)
  have a_smul_y' : a • y' = y := by
    refine Subtype.mk_eq_mk.mp (show (a • ⟨y', y'M⟩ : M) = ⟨y, N_le_M yN⟩ from ?_)
    rw [← b'M.sum_repr ⟨y, N_le_M yN⟩, mk_y', Finset.smul_sum]
    refine Finset.sum_congr rfl fun i _ ↦ ?_
    rw [← mul_smul, ← hc]
    rfl
  -- We found a `y` and an `a`!
  /-
    case neg.intro.intro.intro
    ι : Type u_1
    R : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : Finite ι
    O : Type u_4
    inst✝¹ : AddCommGroup O
    inst✝ : Module R O
    M N : Submodule R O
    b'M : Basis ι R (Subtype fun x => Membership.mem M x)
    N_bot : Ne N Bot.bot
    N_le_M : LE.le N M
    this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
    ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
    a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
    a_mem : Membership.mem (ϕ.submoduleImage N) a
    a_zero : Not (Eq a 0)
    y : O
    yN : Membership.mem N y
    ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
    _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
    c : ι → R
    hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
    val✝ : Fintype ι
    y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
    y'M : Membership.mem M y'
    mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
    a_smul_y' : Eq (HSMul.hSMul a y') y
    ⊢ Exists fun y => And (Membership.mem M y) (Exists fun a => And (Membership.me …
  -/
  refine ⟨y', y'M, a, a_smul_y'.symm ▸ yN, ?_⟩
  have ϕy'_eq : ϕ ⟨y', y'M⟩ = 1 :=
    mul_left_cancel₀ a_zero
      (calc
        a • ϕ ⟨y', y'M⟩ = ϕ ⟨a • y', _⟩ := (ϕ.map_smul a ⟨y', y'M⟩).symm
        _ = ϕ ⟨y, N_le_M yN⟩ := by simp only [a_smul_y']
        _ = a := ϕy_eq
        _ = a * 1 := (mul_one a).symm
        )
  /-
    case neg.intro.intro.intro
    ι : Type u_1
    R : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : Finite ι
    O : Type u_4
    inst✝¹ : AddCommGroup O
    inst✝ : Module R O
    M N : Submodule R O
    b'M : Basis ι R (Subtype fun x => Membership.mem M x)
    N_bot : Ne N Bot.bot
    N_le_M : LE.le N M
    this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
    ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
    a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
    a_mem : Membership.mem (ϕ.submoduleImage N) a
    a_zero : Not (Eq a 0)
    y : O
    yN : Membership.mem N y
    ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
    _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
    c : ι → R
    hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
    val✝ : Fintype ι
    y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
    y'M : Membership.mem M y'
    mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
    a_smul_y' : Eq (HSMul.hSMul a y') y
    ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
    ⊢ Exists fun M' => And (LE.le M' M) (Exists fun N' => And (LE.le N' N) (And (L …
  -/
  have ϕy'_ne_zero : ϕ ⟨y', y'M⟩ ≠ 0 := by simpa only [ϕy'_eq] using one_ne_zero
  -- `M' := ker (ϕ : M → R)` is smaller than `M` and `N' := ker (ϕ : N → R)` is smaller than `N`.
  /-
    case neg.intro.intro.intro
    ι : Type u_1
    R : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : Finite ι
    O : Type u_4
    inst✝¹ : AddCommGroup O
    inst✝ : Module R O
    M N : Submodule R O
    b'M : Basis ι R (Subtype fun x => Membership.mem M x)
    N_bot : Ne N Bot.bot
    N_le_M : LE.le N M
    this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
    ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
    a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
    a_mem : Membership.mem (ϕ.submoduleImage N) a
    a_zero : Not (Eq a 0)
    y : O
    yN : Membership.mem N y
    ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
    _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
    c : ι → R
    hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
    val✝ : Fintype ι
    y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
    y'M : Membership.mem M y'
    mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
    a_smul_y' : Eq (HSMul.hSMul a y') y
    ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
    ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
    ⊢ Exists fun M' => And (LE.le M' M) (Exists fun N' => And (LE.le N' N) (And (L …
  -/
  let M' : Submodule R O := ϕ.ker.map M.subtype
  /-
    case neg.intro.intro.intro
    ι : Type u_1
    R : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : Finite ι
    O : Type u_4
    inst✝¹ : AddCommGroup O
    inst✝ : Module R O
    M N : Submodule R O
    b'M : Basis ι R (Subtype fun x => Membership.mem M x)
    N_bot : Ne N Bot.bot
    N_le_M : LE.le N M
    this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
    ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
    a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
    a_mem : Membership.mem (ϕ.submoduleImage N) a
    a_zero : Not (Eq a 0)
    y : O
    yN : Membership.mem N y
    ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
    _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
    c : ι → R
    hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
    val✝ : Fintype ι
    y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
    y'M : Membership.mem M y'
    mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
    a_smul_y' : Eq (HSMul.hSMul a y') y
    ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
    ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
    M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
    ⊢ Exists fun M' => And (LE.le M' M) (Exists fun N' => And (LE.le N' N) (And (L …
  -/
  let N' : Submodule R O := (ϕ.comp (inclusion N_le_M)).ker.map N.subtype
  /-
    case neg.intro.intro.intro
    ι : Type u_1
    R : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : Finite ι
    O : Type u_4
    inst✝¹ : AddCommGroup O
    inst✝ : Module R O
    M N : Submodule R O
    b'M : Basis ι R (Subtype fun x => Membership.mem M x)
    N_bot : Ne N Bot.bot
    N_le_M : LE.le N M
    this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
    ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
    a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
    a_mem : Membership.mem (ϕ.submoduleImage N) a
    a_zero : Not (Eq a 0)
    y : O
    yN : Membership.mem N y
    ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
    _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
    c : ι → R
    hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
    val✝ : Fintype ι
    y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
    y'M : Membership.mem M y'
    mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
    a_smul_y' : Eq (HSMul.hSMul a y') y
    ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
    ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
    M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
    N' : Submodule R O := Submodule.map N.subtype (LinearMap.ker (ϕ.comp (Submodul …
    ⊢ Exists fun M' => And (LE.le M' M) (Exists fun N' => And (LE.le N' N) (And (L …
  -/
  have M'_le_M : M' ≤ M := M.map_subtype_le (LinearMap.ker ϕ)
  have N'_le_M' : N' ≤ M' := by
    intro x hx
    simp only [N', mem_map, LinearMap.mem_ker] at hx ⊢
    obtain ⟨⟨x, xN⟩, hx, rfl⟩ := hx
    exact ⟨⟨x, N_le_M xN⟩, hx, rfl⟩
  /-
    case neg.intro.intro.intro
    ι : Type u_1
    R : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : Finite ι
    O : Type u_4
    inst✝¹ : AddCommGroup O
    inst✝ : Module R O
    M N : Submodule R O
    b'M : Basis ι R (Subtype fun x => Membership.mem M x)
    N_bot : Ne N Bot.bot
    N_le_M : LE.le N M
    this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
    ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
    a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
    a_mem : Membership.mem (ϕ.submoduleImage N) a
    a_zero : Not (Eq a 0)
    y : O
    yN : Membership.mem N y
    ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
    _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
    c : ι → R
    hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
    val✝ : Fintype ι
    y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
    y'M : Membership.mem M y'
    mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
    a_smul_y' : Eq (HSMul.hSMul a y') y
    ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
    ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
    M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
    N' : Submodule R O := Submodule.map N.subtype (LinearMap.ker (ϕ.comp (Submodul …
    M'_le_M : LE.le M' M
    N'_le_M' : LE.le N' M'
    ⊢ Exists fun M' => And (LE.le M' M) (Exists fun N' => And (LE.le N' N) (And (L …
  -/
  have N'_le_N : N' ≤ N := N.map_subtype_le (LinearMap.ker (ϕ.comp (inclusion N_le_M)))
  -- So fill in those results as well.
  /-
    case neg.intro.intro.intro
    ι : Type u_1
    R : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : Finite ι
    O : Type u_4
    inst✝¹ : AddCommGroup O
    inst✝ : Module R O
    M N : Submodule R O
    b'M : Basis ι R (Subtype fun x => Membership.mem M x)
    N_bot : Ne N Bot.bot
    N_le_M : LE.le N M
    this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
    ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
    a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
    a_mem : Membership.mem (ϕ.submoduleImage N) a
    a_zero : Not (Eq a 0)
    y : O
    yN : Membership.mem N y
    ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
    _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
    c : ι → R
    hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
    val✝ : Fintype ι
    y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
    y'M : Membership.mem M y'
    mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
    a_smul_y' : Eq (HSMul.hSMul a y') y
    ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
    ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
    M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
    N' : Submodule R O := Submodule.map N.subtype (LinearMap.ker (ϕ.comp (Submodul …
    M'_le_M : LE.le M' M
    N'_le_M' : LE.le N' M'
    N'_le_N : LE.le N' N
    ⊢ Exists fun M' => And (LE.le M' M) (Exists fun N' => And (LE.le N' N) (And (L …
  -/
  refine ⟨M', M'_le_M, N', N'_le_N, N'_le_M', ?_⟩
  -- Note that `y'` is orthogonal to `M'`.
  have y'_ortho_M' : ∀ (c : R), ∀ z ∈ M', c • y' + z = 0 → c = 0 := by
    intro c x xM' hc
    obtain ⟨⟨x, xM⟩, hx', rfl⟩ := Submodule.mem_map.mp xM'
    rw [LinearMap.mem_ker] at hx'
    have hc' : (c • ⟨y', y'M⟩ + ⟨x, xM⟩ : M) = 0 := by exact @Subtype.coe_injective O (· ∈ M) _ _ hc
    simpa only [LinearMap.map_add, LinearMap.map_zero, LinearMap.map_smul, smul_eq_mul, add_zero,
      mul_eq_zero, ϕy'_ne_zero, hx', or_false] using congr_arg ϕ hc'
  -- And `a • y'` is orthogonal to `N'`.
  have ay'_ortho_N' : ∀ (c : R), ∀ z ∈ N', c • a • y' + z = 0 → c = 0 := by
    intro c z zN' hc
    refine (mul_eq_zero.mp (y'_ortho_M' (a * c) z (N'_le_M' zN') ?_)).resolve_left a_zero
    rw [mul_comm, mul_smul, hc]
  -- So we can extend a basis for `N'` with `y`
  /-
    case neg.intro.intro.intro
    ι : Type u_1
    R : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : Finite ι
    O : Type u_4
    inst✝¹ : AddCommGroup O
    inst✝ : Module R O
    M N : Submodule R O
    b'M : Basis ι R (Subtype fun x => Membership.mem M x)
    N_bot : Ne N Bot.bot
    N_le_M : LE.le N M
    this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
    ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
    a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
    a_mem : Membership.mem (ϕ.submoduleImage N) a
    a_zero : Not (Eq a 0)
    y : O
    yN : Membership.mem N y
    ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
    _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
    c : ι → R
    hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
    val✝ : Fintype ι
    y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
    y'M : Membership.mem M y'
    mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
    a_smul_y' : Eq (HSMul.hSMul a y') y
    ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
    ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
    M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
    N' : Submodule R O := Submodule.map N.subtype (LinearMap.ker (ϕ.comp (Submodul …
    M'_le_M : LE.le M' M
    N'_le_M' : LE.le N' M'
    N'_le_N : LE.le N' N
    y'_ortho_M' : ∀ (c : R) (z : O), Membership.mem M' z → Eq (HAdd.hAdd (HSMul.hS …
    ay'_ortho_N' : ∀ (c : R) (z : O), Membership.mem N' z → Eq (HAdd.hAdd (HSMul.h …
    ⊢ And (∀ (c : R) (z : O), Membership.mem M' z → Eq (HAdd.hAdd (HSMul.hSMul c y …
  -/
  refine ⟨y'_ortho_M', ay'_ortho_N', fun n' bN' ↦ ⟨?_, ?_⟩⟩
    /-
      case neg.intro.intro.intro.refine_1
      ι : Type u_1
      R : Type u_2
      inst✝⁵ : CommRing R
      inst✝⁴ : IsDomain R
      inst✝³ : IsPrincipalIdealRing R
      inst✝² : Finite ι
      O : Type u_4
      inst✝¹ : AddCommGroup O
      inst✝ : Module R O
      M N : Submodule R O
      b'M : Basis ι R (Subtype fun x => Membership.mem M x)
      N_bot : Ne N Bot.bot
      N_le_M : LE.le N M
      this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
      ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
      ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
      a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
      a_mem : Membership.mem (ϕ.submoduleImage N) a
      a_zero : Not (Eq a 0)
      y : O
      yN : Membership.mem N y
      ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
      _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
      c : ι → R
      hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
      val✝ : Fintype ι
      y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
      y'M : Membership.mem M y'
      mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
      a_smul_y' : Eq (HSMul.hSMul a y') y
      ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
      ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
      M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
      N' : Submodule R O := Submodule.map N.subtype (LinearMap.ker (ϕ.comp (Submodul …
      M'_le_M : LE.le M' M
      N'_le_M' : LE.le N' M'
      N'_le_N : LE.le N' N
      y'_ortho_M' : ∀ (c : R) (z : O), Membership.mem M' z → Eq (HAdd.hAdd (HSMul.hS …
      ay'_ortho_N' : ∀ (c : R) (z : O), Membership.mem N' z → Eq (HAdd.hAdd (HSMul.h …
      n' : Nat
      bN' : Basis (Fin n') R (Subtype fun x => Membership.mem N' x)
      ⊢ Basis (Fin (HAdd.hAdd n' 1)) R (Subtype fun x => Membership.mem N x)
    -/
  · refine Basis.mkFinConsOfLE y yN bN' N'_le_N ?_ ?_
      /-
        case neg.intro.intro.intro.refine_1.refine_1
        ι : Type u_1
        R : Type u_2
        inst✝⁵ : CommRing R
        inst✝⁴ : IsDomain R
        inst✝³ : IsPrincipalIdealRing R
        inst✝² : Finite ι
        O : Type u_4
        inst✝¹ : AddCommGroup O
        inst✝ : Module R O
        M N : Submodule R O
        b'M : Basis ι R (Subtype fun x => Membership.mem M x)
        N_bot : Ne N Bot.bot
        N_le_M : LE.le N M
        this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
        ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
        ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
        a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
        a_mem : Membership.mem (ϕ.submoduleImage N) a
        a_zero : Not (Eq a 0)
        y : O
        yN : Membership.mem N y
        ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
        _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
        c : ι → R
        hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
        val✝ : Fintype ι
        y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
        y'M : Membership.mem M y'
        mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
        a_smul_y' : Eq (HSMul.hSMul a y') y
        ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
        ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
        M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
        N' : Submodule R O := Submodule.map N.subtype (LinearMap.ker (ϕ.comp (Submodul …
        M'_le_M : LE.le M' M
        N'_le_M' : LE.le N' M'
        N'_le_N : LE.le N' N
        y'_ortho_M' : ∀ (c : R) (z : O), Membership.mem M' z → Eq (HAdd.hAdd (HSMul.hS …
        ay'_ortho_N' : ∀ (c : R) (z : O), Membership.mem N' z → Eq (HAdd.hAdd (HSMul.h …
        n' : Nat
        bN' : Basis (Fin n') R (Subtype fun x => Membership.mem N' x)
        ⊢ ∀ (c : R) (x : O), Membership.mem N' x → Eq (HAdd.hAdd (HSMul.hSMul c y) x)  …
      -/
    · intro c z zN' hc
      /-
        case neg.intro.intro.intro.refine_1.refine_1
        ι : Type u_1
        R : Type u_2
        inst✝⁵ : CommRing R
        inst✝⁴ : IsDomain R
        inst✝³ : IsPrincipalIdealRing R
        inst✝² : Finite ι
        O : Type u_4
        inst✝¹ : AddCommGroup O
        inst✝ : Module R O
        M N : Submodule R O
        b'M : Basis ι R (Subtype fun x => Membership.mem M x)
        N_bot : Ne N Bot.bot
        N_le_M : LE.le N M
        this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
        ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
        ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
        a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
        a_mem : Membership.mem (ϕ.submoduleImage N) a
        a_zero : Not (Eq a 0)
        y : O
        yN : Membership.mem N y
        ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
        _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
        c✝ : ι → R
        hc✝ : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c✝ i))
        val✝ : Fintype ι
        y' : O := Finset.univ.sum fun i => HSMul.hSMul (c✝ i) ↑(b'M i)
        y'M : Membership.mem M y'
        mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c✝ i) (b'M i))
        a_smul_y' : Eq (HSMul.hSMul a y') y
        ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
        ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
        M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
        N' : Submodule R O := Submodule.map N.subtype (LinearMap.ker (ϕ.comp (Submodul …
        M'_le_M : LE.le M' M
        N'_le_M' : LE.le N' M'
        N'_le_N : LE.le N' N
        y'_ortho_M' : ∀ (c : R) (z : O), Membership.mem M' z → Eq (HAdd.hAdd (HSMul.hS …
        ay'_ortho_N' : ∀ (c : R) (z : O), Membership.mem N' z → Eq (HAdd.hAdd (HSMul.h …
        n' : Nat
        bN' : Basis (Fin n') R (Subtype fun x => Membership.mem N' x)
        c : R
        z : O
        zN' : Membership.mem N' z
        hc : Eq (HAdd.hAdd (HSMul.hSMul c y) z) 0
        ⊢ Eq c 0
      -/
      refine ay'_ortho_N' c z zN' ?_
      /-
        case neg.intro.intro.intro.refine_1.refine_1
        ι : Type u_1
        R : Type u_2
        inst✝⁵ : CommRing R
        inst✝⁴ : IsDomain R
        inst✝³ : IsPrincipalIdealRing R
        inst✝² : Finite ι
        O : Type u_4
        inst✝¹ : AddCommGroup O
        inst✝ : Module R O
        M N : Submodule R O
        b'M : Basis ι R (Subtype fun x => Membership.mem M x)
        N_bot : Ne N Bot.bot
        N_le_M : LE.le N M
        this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
        ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
        ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
        a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
        a_mem : Membership.mem (ϕ.submoduleImage N) a
        a_zero : Not (Eq a 0)
        y : O
        yN : Membership.mem N y
        ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
        _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
        c✝ : ι → R
        hc✝ : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c✝ i))
        val✝ : Fintype ι
        y' : O := Finset.univ.sum fun i => HSMul.hSMul (c✝ i) ↑(b'M i)
        y'M : Membership.mem M y'
        mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c✝ i) (b'M i))
        a_smul_y' : Eq (HSMul.hSMul a y') y
        ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
        ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
        M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
        N' : Submodule R O := Submodule.map N.subtype (LinearMap.ker (ϕ.comp (Submodul …
        M'_le_M : LE.le M' M
        N'_le_M' : LE.le N' M'
        N'_le_N : LE.le N' N
        y'_ortho_M' : ∀ (c : R) (z : O), Membership.mem M' z → Eq (HAdd.hAdd (HSMul.hS …
        ay'_ortho_N' : ∀ (c : R) (z : O), Membership.mem N' z → Eq (HAdd.hAdd (HSMul.h …
        n' : Nat
        bN' : Basis (Fin n') R (Subtype fun x => Membership.mem N' x)
        c : R
        z : O
        zN' : Membership.mem N' z
        hc : Eq (HAdd.hAdd (HSMul.hSMul c y) z) 0
        ⊢ Eq (HAdd.hAdd (HSMul.hSMul c (HSMul.hSMul a y')) z) 0
      -/
      rwa [← a_smul_y'] at hc
      /-
        🎉 no goals
      -/
      /-
        case neg.intro.intro.intro.refine_1.refine_2
        ι : Type u_1
        R : Type u_2
        inst✝⁵ : CommRing R
        inst✝⁴ : IsDomain R
        inst✝³ : IsPrincipalIdealRing R
        inst✝² : Finite ι
        O : Type u_4
        inst✝¹ : AddCommGroup O
        inst✝ : Module R O
        M N : Submodule R O
        b'M : Basis ι R (Subtype fun x => Membership.mem M x)
        N_bot : Ne N Bot.bot
        N_le_M : LE.le N M
        this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
        ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
        ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
        a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
        a_mem : Membership.mem (ϕ.submoduleImage N) a
        a_zero : Not (Eq a 0)
        y : O
        yN : Membership.mem N y
        ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
        _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
        c : ι → R
        hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
        val✝ : Fintype ι
        y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
        y'M : Membership.mem M y'
        mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
        a_smul_y' : Eq (HSMul.hSMul a y') y
        ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
        ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
        M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
        N' : Submodule R O := Submodule.map N.subtype (LinearMap.ker (ϕ.comp (Submodul …
        M'_le_M : LE.le M' M
        N'_le_M' : LE.le N' M'
        N'_le_N : LE.le N' N
        y'_ortho_M' : ∀ (c : R) (z : O), Membership.mem M' z → Eq (HAdd.hAdd (HSMul.hS …
        ay'_ortho_N' : ∀ (c : R) (z : O), Membership.mem N' z → Eq (HAdd.hAdd (HSMul.h …
        n' : Nat
        bN' : Basis (Fin n') R (Subtype fun x => Membership.mem N' x)
        ⊢ ∀ (z : O), Membership.mem N z → Exists fun c => Membership.mem N' (HAdd.hAdd …
      -/
    · intro z zN
      /-
        case neg.intro.intro.intro.refine_1.refine_2
        ι : Type u_1
        R : Type u_2
        inst✝⁵ : CommRing R
        inst✝⁴ : IsDomain R
        inst✝³ : IsPrincipalIdealRing R
        inst✝² : Finite ι
        O : Type u_4
        inst✝¹ : AddCommGroup O
        inst✝ : Module R O
        M N : Submodule R O
        b'M : Basis ι R (Subtype fun x => Membership.mem M x)
        N_bot : Ne N Bot.bot
        N_le_M : LE.le N M
        this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
        ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
        ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
        a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
        a_mem : Membership.mem (ϕ.submoduleImage N) a
        a_zero : Not (Eq a 0)
        y : O
        yN : Membership.mem N y
        ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
        _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
        c : ι → R
        hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
        val✝ : Fintype ι
        y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
        y'M : Membership.mem M y'
        mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
        a_smul_y' : Eq (HSMul.hSMul a y') y
        ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
        ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
        M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
        N' : Submodule R O := Submodule.map N.subtype (LinearMap.ker (ϕ.comp (Submodul …
        M'_le_M : LE.le M' M
        N'_le_M' : LE.le N' M'
        N'_le_N : LE.le N' N
        y'_ortho_M' : ∀ (c : R) (z : O), Membership.mem M' z → Eq (HAdd.hAdd (HSMul.hS …
        ay'_ortho_N' : ∀ (c : R) (z : O), Membership.mem N' z → Eq (HAdd.hAdd (HSMul.h …
        n' : Nat
        bN' : Basis (Fin n') R (Subtype fun x => Membership.mem N' x)
        z : O
        zN : Membership.mem N z
        ⊢ Exists fun c => Membership.mem N' (HAdd.hAdd z (HSMul.hSMul c y))
      -/
      obtain ⟨b, hb⟩ : _ ∣ ϕ ⟨z, N_le_M zN⟩ := generator_submoduleImage_dvd_of_mem N_le_M ϕ zN
      /-
        case neg.intro.intro.intro.refine_1.refine_2.intro
        ι : Type u_1
        R : Type u_2
        inst✝⁵ : CommRing R
        inst✝⁴ : IsDomain R
        inst✝³ : IsPrincipalIdealRing R
        inst✝² : Finite ι
        O : Type u_4
        inst✝¹ : AddCommGroup O
        inst✝ : Module R O
        M N : Submodule R O
        b'M : Basis ι R (Subtype fun x => Membership.mem M x)
        N_bot : Ne N Bot.bot
        N_le_M : LE.le N M
        this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
        ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
        ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
        a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
        a_mem : Membership.mem (ϕ.submoduleImage N) a
        a_zero : Not (Eq a 0)
        y : O
        yN : Membership.mem N y
        ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
        _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
        c : ι → R
        hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
        val✝ : Fintype ι
        y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
        y'M : Membership.mem M y'
        mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
        a_smul_y' : Eq (HSMul.hSMul a y') y
        ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
        ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
        M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
        N' : Submodule R O := Submodule.map N.subtype (LinearMap.ker (ϕ.comp (Submodul …
        M'_le_M : LE.le M' M
        N'_le_M' : LE.le N' M'
        N'_le_N : LE.le N' N
        y'_ortho_M' : ∀ (c : R) (z : O), Membership.mem M' z → Eq (HAdd.hAdd (HSMul.hS …
        ay'_ortho_N' : ∀ (c : R) (z : O), Membership.mem N' z → Eq (HAdd.hAdd (HSMul.h …
        n' : Nat
        bN' : Basis (Fin n') R (Subtype fun x => Membership.mem N' x)
        z : O
        zN : Membership.mem N z
        b : R
        hb : Eq (ϕ ⟨z, ⋯⟩) (HMul.hMul (Submodule.IsPrincipal.generator (ϕ.submoduleIma …
        ⊢ Exists fun c => Membership.mem N' (HAdd.hAdd z (HSMul.hSMul c y))
      -/
      refine ⟨-b, Submodule.mem_map.mpr ⟨⟨_, N.sub_mem zN (N.smul_mem b yN)⟩, ?_, ?_⟩⟩
        /-
          case neg.intro.intro.intro.refine_1.refine_2.intro.refine_1
          ι : Type u_1
          R : Type u_2
          inst✝⁵ : CommRing R
          inst✝⁴ : IsDomain R
          inst✝³ : IsPrincipalIdealRing R
          inst✝² : Finite ι
          O : Type u_4
          inst✝¹ : AddCommGroup O
          inst✝ : Module R O
          M N : Submodule R O
          b'M : Basis ι R (Subtype fun x => Membership.mem M x)
          N_bot : Ne N Bot.bot
          N_le_M : LE.le N M
          this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
          ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
          ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
          a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
          a_mem : Membership.mem (ϕ.submoduleImage N) a
          a_zero : Not (Eq a 0)
          y : O
          yN : Membership.mem N y
          ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
          _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
          c : ι → R
          hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
          val✝ : Fintype ι
          y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
          y'M : Membership.mem M y'
          mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
          a_smul_y' : Eq (HSMul.hSMul a y') y
          ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
          ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
          M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
          N' : Submodule R O := Submodule.map N.subtype (LinearMap.ker (ϕ.comp (Submodul …
          M'_le_M : LE.le M' M
          N'_le_M' : LE.le N' M'
          N'_le_N : LE.le N' N
          y'_ortho_M' : ∀ (c : R) (z : O), Membership.mem M' z → Eq (HAdd.hAdd (HSMul.hS …
          ay'_ortho_N' : ∀ (c : R) (z : O), Membership.mem N' z → Eq (HAdd.hAdd (HSMul.h …
          n' : Nat
          bN' : Basis (Fin n') R (Subtype fun x => Membership.mem N' x)
          z : O
          zN : Membership.mem N z
          b : R
          hb : Eq (ϕ ⟨z, ⋯⟩) (HMul.hMul (Submodule.IsPrincipal.generator (ϕ.submoduleIma …
          ⊢ Membership.mem (LinearMap.ker (ϕ.comp (Submodule.inclusion N_le_M))) ⟨HSub.h …
        -/
      · refine LinearMap.mem_ker.mpr (show ϕ (⟨z, N_le_M zN⟩ - b • ⟨y, N_le_M yN⟩) = 0 from ?_)
        /-
          case neg.intro.intro.intro.refine_1.refine_2.intro.refine_1
          ι : Type u_1
          R : Type u_2
          inst✝⁵ : CommRing R
          inst✝⁴ : IsDomain R
          inst✝³ : IsPrincipalIdealRing R
          inst✝² : Finite ι
          O : Type u_4
          inst✝¹ : AddCommGroup O
          inst✝ : Module R O
          M N : Submodule R O
          b'M : Basis ι R (Subtype fun x => Membership.mem M x)
          N_bot : Ne N Bot.bot
          N_le_M : LE.le N M
          this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
          ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
          ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
          a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
          a_mem : Membership.mem (ϕ.submoduleImage N) a
          a_zero : Not (Eq a 0)
          y : O
          yN : Membership.mem N y
          ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
          _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
          c : ι → R
          hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
          val✝ : Fintype ι
          y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
          y'M : Membership.mem M y'
          mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
          a_smul_y' : Eq (HSMul.hSMul a y') y
          ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
          ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
          M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
          N' : Submodule R O := Submodule.map N.subtype (LinearMap.ker (ϕ.comp (Submodul …
          M'_le_M : LE.le M' M
          N'_le_M' : LE.le N' M'
          N'_le_N : LE.le N' N
          y'_ortho_M' : ∀ (c : R) (z : O), Membership.mem M' z → Eq (HAdd.hAdd (HSMul.hS …
          ay'_ortho_N' : ∀ (c : R) (z : O), Membership.mem N' z → Eq (HAdd.hAdd (HSMul.h …
          n' : Nat
          bN' : Basis (Fin n') R (Subtype fun x => Membership.mem N' x)
          z : O
          zN : Membership.mem N z
          b : R
          hb : Eq (ϕ ⟨z, ⋯⟩) (HMul.hMul (Submodule.IsPrincipal.generator (ϕ.submoduleIma …
          ⊢ Eq (ϕ (HSub.hSub ⟨z, ⋯⟩ (HSMul.hSMul b ⟨y, ⋯⟩))) 0
        -/
        rw [LinearMap.map_sub, LinearMap.map_smul, hb, ϕy_eq, smul_eq_mul, mul_comm, sub_self]
        /-
          🎉 no goals
        -/
        /-
          case neg.intro.intro.intro.refine_1.refine_2.intro.refine_2
          ι : Type u_1
          R : Type u_2
          inst✝⁵ : CommRing R
          inst✝⁴ : IsDomain R
          inst✝³ : IsPrincipalIdealRing R
          inst✝² : Finite ι
          O : Type u_4
          inst✝¹ : AddCommGroup O
          inst✝ : Module R O
          M N : Submodule R O
          b'M : Basis ι R (Subtype fun x => Membership.mem M x)
          N_bot : Ne N Bot.bot
          N_le_M : LE.le N M
          this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
          ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
          ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
          a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
          a_mem : Membership.mem (ϕ.submoduleImage N) a
          a_zero : Not (Eq a 0)
          y : O
          yN : Membership.mem N y
          ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
          _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
          c : ι → R
          hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
          val✝ : Fintype ι
          y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
          y'M : Membership.mem M y'
          mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
          a_smul_y' : Eq (HSMul.hSMul a y') y
          ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
          ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
          M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
          N' : Submodule R O := Submodule.map N.subtype (LinearMap.ker (ϕ.comp (Submodul …
          M'_le_M : LE.le M' M
          N'_le_M' : LE.le N' M'
          N'_le_N : LE.le N' N
          y'_ortho_M' : ∀ (c : R) (z : O), Membership.mem M' z → Eq (HAdd.hAdd (HSMul.hS …
          ay'_ortho_N' : ∀ (c : R) (z : O), Membership.mem N' z → Eq (HAdd.hAdd (HSMul.h …
          n' : Nat
          bN' : Basis (Fin n') R (Subtype fun x => Membership.mem N' x)
          z : O
          zN : Membership.mem N z
          b : R
          hb : Eq (ϕ ⟨z, ⋯⟩) (HMul.hMul (Submodule.IsPrincipal.generator (ϕ.submoduleIma …
          ⊢ Eq (N.subtype ⟨HSub.hSub z (HSMul.hSMul b y), ⋯⟩) (HAdd.hAdd z (HSMul.hSMul  …
        -/
      · simp only [sub_eq_add_neg, neg_smul, coe_subtype]
        /-
          🎉 no goals
        -/
  -- And extend a basis for `M'` with `y'`
  /-
    case neg.intro.intro.intro.refine_2
    ι : Type u_1
    R : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : Finite ι
    O : Type u_4
    inst✝¹ : AddCommGroup O
    inst✝ : Module R O
    M N : Submodule R O
    b'M : Basis ι R (Subtype fun x => Membership.mem M x)
    N_bot : Ne N Bot.bot
    N_le_M : LE.le N M
    this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
    ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
    a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
    a_mem : Membership.mem (ϕ.submoduleImage N) a
    a_zero : Not (Eq a 0)
    y : O
    yN : Membership.mem N y
    ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
    _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
    c : ι → R
    hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
    val✝ : Fintype ι
    y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
    y'M : Membership.mem M y'
    mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
    a_smul_y' : Eq (HSMul.hSMul a y') y
    ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
    ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
    M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
    N' : Submodule R O := Submodule.map N.subtype (LinearMap.ker (ϕ.comp (Submodul …
    M'_le_M : LE.le M' M
    N'_le_M' : LE.le N' M'
    N'_le_N : LE.le N' N
    y'_ortho_M' : ∀ (c : R) (z : O), Membership.mem M' z → Eq (HAdd.hAdd (HSMul.hS …
    ay'_ortho_N' : ∀ (c : R) (z : O), Membership.mem N' z → Eq (HAdd.hAdd (HSMul.h …
    n' : Nat
    bN' : Basis (Fin n') R (Subtype fun x => Membership.mem N' x)
    ⊢ ∀ (m' : Nat) (hn'm' : LE.le n' m') (bM' : Basis (Fin m') R (Subtype fun x => …
  -/
  intro m' hn'm' bM'
  /-
    case neg.intro.intro.intro.refine_2
    ι : Type u_1
    R : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : Finite ι
    O : Type u_4
    inst✝¹ : AddCommGroup O
    inst✝ : Module R O
    M N : Submodule R O
    b'M : Basis ι R (Subtype fun x => Membership.mem M x)
    N_bot : Ne N Bot.bot
    N_le_M : LE.le N M
    this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
    ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
    a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
    a_mem : Membership.mem (ϕ.submoduleImage N) a
    a_zero : Not (Eq a 0)
    y : O
    yN : Membership.mem N y
    ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
    _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
    c : ι → R
    hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
    val✝ : Fintype ι
    y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
    y'M : Membership.mem M y'
    mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
    a_smul_y' : Eq (HSMul.hSMul a y') y
    ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
    ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
    M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
    N' : Submodule R O := Submodule.map N.subtype (LinearMap.ker (ϕ.comp (Submodul …
    M'_le_M : LE.le M' M
    N'_le_M' : LE.le N' M'
    N'_le_N : LE.le N' N
    y'_ortho_M' : ∀ (c : R) (z : O), Membership.mem M' z → Eq (HAdd.hAdd (HSMul.hS …
    ay'_ortho_N' : ∀ (c : R) (z : O), Membership.mem N' z → Eq (HAdd.hAdd (HSMul.h …
    n' : Nat
    bN' : Basis (Fin n') R (Subtype fun x => Membership.mem N' x)
    m' : Nat
    hn'm' : LE.le n' m'
    bM' : Basis (Fin m') R (Subtype fun x => Membership.mem M' x)
    ⊢ Exists fun hnm => Exists fun bM => ∀ (as : Fin n' → R), (∀ (i : Fin n'), Eq  …
  -/
  refine ⟨Nat.succ_le_succ hn'm', ?_, ?_⟩
    /-
      case neg.intro.intro.intro.refine_2.refine_1
      ι : Type u_1
      R : Type u_2
      inst✝⁵ : CommRing R
      inst✝⁴ : IsDomain R
      inst✝³ : IsPrincipalIdealRing R
      inst✝² : Finite ι
      O : Type u_4
      inst✝¹ : AddCommGroup O
      inst✝ : Module R O
      M N : Submodule R O
      b'M : Basis ι R (Subtype fun x => Membership.mem M x)
      N_bot : Ne N Bot.bot
      N_le_M : LE.le N M
      this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
      ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
      ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
      a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
      a_mem : Membership.mem (ϕ.submoduleImage N) a
      a_zero : Not (Eq a 0)
      y : O
      yN : Membership.mem N y
      ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
      _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
      c : ι → R
      hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
      val✝ : Fintype ι
      y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
      y'M : Membership.mem M y'
      mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
      a_smul_y' : Eq (HSMul.hSMul a y') y
      ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
      ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
      M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
      N' : Submodule R O := Submodule.map N.subtype (LinearMap.ker (ϕ.comp (Submodul …
      M'_le_M : LE.le M' M
      N'_le_M' : LE.le N' M'
      N'_le_N : LE.le N' N
      y'_ortho_M' : ∀ (c : R) (z : O), Membership.mem M' z → Eq (HAdd.hAdd (HSMul.hS …
      ay'_ortho_N' : ∀ (c : R) (z : O), Membership.mem N' z → Eq (HAdd.hAdd (HSMul.h …
      n' : Nat
      bN' : Basis (Fin n') R (Subtype fun x => Membership.mem N' x)
      m' : Nat
      hn'm' : LE.le n' m'
      bM' : Basis (Fin m') R (Subtype fun x => Membership.mem M' x)
      ⊢ Basis (Fin (HAdd.hAdd m' 1)) R (Subtype fun x => Membership.mem M x)
    -/
  · refine Basis.mkFinConsOfLE y' y'M bM' M'_le_M y'_ortho_M' ?_
    /-
      case neg.intro.intro.intro.refine_2.refine_1
      ι : Type u_1
      R : Type u_2
      inst✝⁵ : CommRing R
      inst✝⁴ : IsDomain R
      inst✝³ : IsPrincipalIdealRing R
      inst✝² : Finite ι
      O : Type u_4
      inst✝¹ : AddCommGroup O
      inst✝ : Module R O
      M N : Submodule R O
      b'M : Basis ι R (Subtype fun x => Membership.mem M x)
      N_bot : Ne N Bot.bot
      N_le_M : LE.le N M
      this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
      ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
      ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
      a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
      a_mem : Membership.mem (ϕ.submoduleImage N) a
      a_zero : Not (Eq a 0)
      y : O
      yN : Membership.mem N y
      ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
      _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
      c : ι → R
      hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
      val✝ : Fintype ι
      y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
      y'M : Membership.mem M y'
      mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
      a_smul_y' : Eq (HSMul.hSMul a y') y
      ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
      ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
      M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
      N' : Submodule R O := Submodule.map N.subtype (LinearMap.ker (ϕ.comp (Submodul …
      M'_le_M : LE.le M' M
      N'_le_M' : LE.le N' M'
      N'_le_N : LE.le N' N
      y'_ortho_M' : ∀ (c : R) (z : O), Membership.mem M' z → Eq (HAdd.hAdd (HSMul.hS …
      ay'_ortho_N' : ∀ (c : R) (z : O), Membership.mem N' z → Eq (HAdd.hAdd (HSMul.h …
      n' : Nat
      bN' : Basis (Fin n') R (Subtype fun x => Membership.mem N' x)
      m' : Nat
      hn'm' : LE.le n' m'
      bM' : Basis (Fin m') R (Subtype fun x => Membership.mem M' x)
      ⊢ ∀ (z : O), Membership.mem M z → Exists fun c => Membership.mem M' (HAdd.hAdd …
    -/
    intro z zM
    /-
      case neg.intro.intro.intro.refine_2.refine_1
      ι : Type u_1
      R : Type u_2
      inst✝⁵ : CommRing R
      inst✝⁴ : IsDomain R
      inst✝³ : IsPrincipalIdealRing R
      inst✝² : Finite ι
      O : Type u_4
      inst✝¹ : AddCommGroup O
      inst✝ : Module R O
      M N : Submodule R O
      b'M : Basis ι R (Subtype fun x => Membership.mem M x)
      N_bot : Ne N Bot.bot
      N_le_M : LE.le N M
      this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
      ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
      ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
      a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
      a_mem : Membership.mem (ϕ.submoduleImage N) a
      a_zero : Not (Eq a 0)
      y : O
      yN : Membership.mem N y
      ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
      _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
      c : ι → R
      hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
      val✝ : Fintype ι
      y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
      y'M : Membership.mem M y'
      mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
      a_smul_y' : Eq (HSMul.hSMul a y') y
      ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
      ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
      M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
      N' : Submodule R O := Submodule.map N.subtype (LinearMap.ker (ϕ.comp (Submodul …
      M'_le_M : LE.le M' M
      N'_le_M' : LE.le N' M'
      N'_le_N : LE.le N' N
      y'_ortho_M' : ∀ (c : R) (z : O), Membership.mem M' z → Eq (HAdd.hAdd (HSMul.hS …
      ay'_ortho_N' : ∀ (c : R) (z : O), Membership.mem N' z → Eq (HAdd.hAdd (HSMul.h …
      n' : Nat
      bN' : Basis (Fin n') R (Subtype fun x => Membership.mem N' x)
      m' : Nat
      hn'm' : LE.le n' m'
      bM' : Basis (Fin m') R (Subtype fun x => Membership.mem M' x)
      z : O
      zM : Membership.mem M z
      ⊢ Exists fun c => Membership.mem M' (HAdd.hAdd z (HSMul.hSMul c y'))
    -/
    refine ⟨-ϕ ⟨z, zM⟩, ⟨⟨z, zM⟩ - ϕ ⟨z, zM⟩ • ⟨y', y'M⟩, LinearMap.mem_ker.mpr ?_, ?_⟩⟩
      /-
        case neg.intro.intro.intro.refine_2.refine_1.refine_1
        ι : Type u_1
        R : Type u_2
        inst✝⁵ : CommRing R
        inst✝⁴ : IsDomain R
        inst✝³ : IsPrincipalIdealRing R
        inst✝² : Finite ι
        O : Type u_4
        inst✝¹ : AddCommGroup O
        inst✝ : Module R O
        M N : Submodule R O
        b'M : Basis ι R (Subtype fun x => Membership.mem M x)
        N_bot : Ne N Bot.bot
        N_le_M : LE.le N M
        this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
        ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
        ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
        a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
        a_mem : Membership.mem (ϕ.submoduleImage N) a
        a_zero : Not (Eq a 0)
        y : O
        yN : Membership.mem N y
        ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
        _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
        c : ι → R
        hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
        val✝ : Fintype ι
        y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
        y'M : Membership.mem M y'
        mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
        a_smul_y' : Eq (HSMul.hSMul a y') y
        ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
        ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
        M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
        N' : Submodule R O := Submodule.map N.subtype (LinearMap.ker (ϕ.comp (Submodul …
        M'_le_M : LE.le M' M
        N'_le_M' : LE.le N' M'
        N'_le_N : LE.le N' N
        y'_ortho_M' : ∀ (c : R) (z : O), Membership.mem M' z → Eq (HAdd.hAdd (HSMul.hS …
        ay'_ortho_N' : ∀ (c : R) (z : O), Membership.mem N' z → Eq (HAdd.hAdd (HSMul.h …
        n' : Nat
        bN' : Basis (Fin n') R (Subtype fun x => Membership.mem N' x)
        m' : Nat
        hn'm' : LE.le n' m'
        bM' : Basis (Fin m') R (Subtype fun x => Membership.mem M' x)
        z : O
        zM : Membership.mem M z
        ⊢ Eq (ϕ (HSub.hSub ⟨z, zM⟩ (HSMul.hSMul (ϕ ⟨z, zM⟩) ⟨y', y'M⟩))) 0
      -/
    · rw [LinearMap.map_sub, LinearMap.map_smul, ϕy'_eq, smul_eq_mul, mul_one, sub_self]
      /-
        🎉 no goals
      -/
      /-
        case neg.intro.intro.intro.refine_2.refine_1.refine_2
        ι : Type u_1
        R : Type u_2
        inst✝⁵ : CommRing R
        inst✝⁴ : IsDomain R
        inst✝³ : IsPrincipalIdealRing R
        inst✝² : Finite ι
        O : Type u_4
        inst✝¹ : AddCommGroup O
        inst✝ : Module R O
        M N : Submodule R O
        b'M : Basis ι R (Subtype fun x => Membership.mem M x)
        N_bot : Ne N Bot.bot
        N_le_M : LE.le N M
        this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
        ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
        ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
        a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
        a_mem : Membership.mem (ϕ.submoduleImage N) a
        a_zero : Not (Eq a 0)
        y : O
        yN : Membership.mem N y
        ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
        _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
        c : ι → R
        hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
        val✝ : Fintype ι
        y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
        y'M : Membership.mem M y'
        mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
        a_smul_y' : Eq (HSMul.hSMul a y') y
        ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
        ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
        M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
        N' : Submodule R O := Submodule.map N.subtype (LinearMap.ker (ϕ.comp (Submodul …
        M'_le_M : LE.le M' M
        N'_le_M' : LE.le N' M'
        N'_le_N : LE.le N' N
        y'_ortho_M' : ∀ (c : R) (z : O), Membership.mem M' z → Eq (HAdd.hAdd (HSMul.hS …
        ay'_ortho_N' : ∀ (c : R) (z : O), Membership.mem N' z → Eq (HAdd.hAdd (HSMul.h …
        n' : Nat
        bN' : Basis (Fin n') R (Subtype fun x => Membership.mem N' x)
        m' : Nat
        hn'm' : LE.le n' m'
        bM' : Basis (Fin m') R (Subtype fun x => Membership.mem M' x)
        z : O
        zM : Membership.mem M z
        ⊢ Eq (M.subtype (HSub.hSub ⟨z, zM⟩ (HSMul.hSMul (ϕ ⟨z, zM⟩) ⟨y', y'M⟩))) (HAdd …
      -/
    · rw [LinearMap.map_sub, LinearMap.map_smul, sub_eq_add_neg, neg_smul]
      /-
        case neg.intro.intro.intro.refine_2.refine_1.refine_2
        ι : Type u_1
        R : Type u_2
        inst✝⁵ : CommRing R
        inst✝⁴ : IsDomain R
        inst✝³ : IsPrincipalIdealRing R
        inst✝² : Finite ι
        O : Type u_4
        inst✝¹ : AddCommGroup O
        inst✝ : Module R O
        M N : Submodule R O
        b'M : Basis ι R (Subtype fun x => Membership.mem M x)
        N_bot : Ne N Bot.bot
        N_le_M : LE.le N M
        this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
        ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
        ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
        a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
        a_mem : Membership.mem (ϕ.submoduleImage N) a
        a_zero : Not (Eq a 0)
        y : O
        yN : Membership.mem N y
        ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
        _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
        c : ι → R
        hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
        val✝ : Fintype ι
        y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
        y'M : Membership.mem M y'
        mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
        a_smul_y' : Eq (HSMul.hSMul a y') y
        ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
        ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
        M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
        N' : Submodule R O := Submodule.map N.subtype (LinearMap.ker (ϕ.comp (Submodul …
        M'_le_M : LE.le M' M
        N'_le_M' : LE.le N' M'
        N'_le_N : LE.le N' N
        y'_ortho_M' : ∀ (c : R) (z : O), Membership.mem M' z → Eq (HAdd.hAdd (HSMul.hS …
        ay'_ortho_N' : ∀ (c : R) (z : O), Membership.mem N' z → Eq (HAdd.hAdd (HSMul.h …
        n' : Nat
        bN' : Basis (Fin n') R (Subtype fun x => Membership.mem N' x)
        m' : Nat
        hn'm' : LE.le n' m'
        bM' : Basis (Fin m') R (Subtype fun x => Membership.mem M' x)
        z : O
        zM : Membership.mem M z
        ⊢ Eq (HAdd.hAdd (M.subtype ⟨z, zM⟩) (Neg.neg (HSMul.hSMul (ϕ ⟨z, zM⟩) (M.subty …
      -/
      rfl
      /-
        🎉 no goals
      -/
  -- It remains to show the extended bases are compatible with each other.
  /-
    case neg.intro.intro.intro.refine_2.refine_2
    ι : Type u_1
    R : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : Finite ι
    O : Type u_4
    inst✝¹ : AddCommGroup O
    inst✝ : Module R O
    M N : Submodule R O
    b'M : Basis ι R (Subtype fun x => Membership.mem M x)
    N_bot : Ne N Bot.bot
    N_le_M : LE.le N M
    this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
    ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
    a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
    a_mem : Membership.mem (ϕ.submoduleImage N) a
    a_zero : Not (Eq a 0)
    y : O
    yN : Membership.mem N y
    ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
    _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
    c : ι → R
    hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
    val✝ : Fintype ι
    y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
    y'M : Membership.mem M y'
    mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
    a_smul_y' : Eq (HSMul.hSMul a y') y
    ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
    ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
    M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
    N' : Submodule R O := Submodule.map N.subtype (LinearMap.ker (ϕ.comp (Submodul …
    M'_le_M : LE.le M' M
    N'_le_M' : LE.le N' M'
    N'_le_N : LE.le N' N
    y'_ortho_M' : ∀ (c : R) (z : O), Membership.mem M' z → Eq (HAdd.hAdd (HSMul.hS …
    ay'_ortho_N' : ∀ (c : R) (z : O), Membership.mem N' z → Eq (HAdd.hAdd (HSMul.h …
    n' : Nat
    bN' : Basis (Fin n') R (Subtype fun x => Membership.mem N' x)
    m' : Nat
    hn'm' : LE.le n' m'
    bM' : Basis (Fin m') R (Subtype fun x => Membership.mem M' x)
    ⊢ ∀ (as : Fin n' → R), (∀ (i : Fin n'), Eq (↑(bN' i)) (HSMul.hSMul (as i) ↑(bM …
  -/
  intro as h
  /-
    case neg.intro.intro.intro.refine_2.refine_2
    ι : Type u_1
    R : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : Finite ι
    O : Type u_4
    inst✝¹ : AddCommGroup O
    inst✝ : Module R O
    M N : Submodule R O
    b'M : Basis ι R (Subtype fun x => Membership.mem M x)
    N_bot : Ne N Bot.bot
    N_le_M : LE.le N M
    this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
    ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
    a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
    a_mem : Membership.mem (ϕ.submoduleImage N) a
    a_zero : Not (Eq a 0)
    y : O
    yN : Membership.mem N y
    ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
    _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
    c : ι → R
    hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
    val✝ : Fintype ι
    y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
    y'M : Membership.mem M y'
    mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
    a_smul_y' : Eq (HSMul.hSMul a y') y
    ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
    ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
    M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
    N' : Submodule R O := Submodule.map N.subtype (LinearMap.ker (ϕ.comp (Submodul …
    M'_le_M : LE.le M' M
    N'_le_M' : LE.le N' M'
    N'_le_N : LE.le N' N
    y'_ortho_M' : ∀ (c : R) (z : O), Membership.mem M' z → Eq (HAdd.hAdd (HSMul.hS …
    ay'_ortho_N' : ∀ (c : R) (z : O), Membership.mem N' z → Eq (HAdd.hAdd (HSMul.h …
    n' : Nat
    bN' : Basis (Fin n') R (Subtype fun x => Membership.mem N' x)
    m' : Nat
    hn'm' : LE.le n' m'
    bM' : Basis (Fin m') R (Subtype fun x => Membership.mem M' x)
    as : Fin n' → R
    h : ∀ (i : Fin n'), Eq (↑(bN' i)) (HSMul.hSMul (as i) ↑(bM' (Fin.castLE hn'm'  …
    ⊢ Exists fun as' => ∀ (i : Fin (HAdd.hAdd n' 1)), Eq (↑((Basis.mkFinConsOfLE y …
  -/
  refine ⟨Fin.cons a as, ?_⟩
  /-
    case neg.intro.intro.intro.refine_2.refine_2
    ι : Type u_1
    R : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : Finite ι
    O : Type u_4
    inst✝¹ : AddCommGroup O
    inst✝ : Module R O
    M N : Submodule R O
    b'M : Basis ι R (Subtype fun x => Membership.mem M x)
    N_bot : Ne N Bot.bot
    N_le_M : LE.le N M
    this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
    ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
    a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
    a_mem : Membership.mem (ϕ.submoduleImage N) a
    a_zero : Not (Eq a 0)
    y : O
    yN : Membership.mem N y
    ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
    _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
    c : ι → R
    hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
    val✝ : Fintype ι
    y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
    y'M : Membership.mem M y'
    mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
    a_smul_y' : Eq (HSMul.hSMul a y') y
    ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
    ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
    M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
    N' : Submodule R O := Submodule.map N.subtype (LinearMap.ker (ϕ.comp (Submodul …
    M'_le_M : LE.le M' M
    N'_le_M' : LE.le N' M'
    N'_le_N : LE.le N' N
    y'_ortho_M' : ∀ (c : R) (z : O), Membership.mem M' z → Eq (HAdd.hAdd (HSMul.hS …
    ay'_ortho_N' : ∀ (c : R) (z : O), Membership.mem N' z → Eq (HAdd.hAdd (HSMul.h …
    n' : Nat
    bN' : Basis (Fin n') R (Subtype fun x => Membership.mem N' x)
    m' : Nat
    hn'm' : LE.le n' m'
    bM' : Basis (Fin m') R (Subtype fun x => Membership.mem M' x)
    as : Fin n' → R
    h : ∀ (i : Fin n'), Eq (↑(bN' i)) (HSMul.hSMul (as i) ↑(bM' (Fin.castLE hn'm'  …
    ⊢ ∀ (i : Fin (HAdd.hAdd n' 1)), Eq (↑((Basis.mkFinConsOfLE y yN bN' N'_le_N ⋯  …
  -/
  intro i
  /-
    case neg.intro.intro.intro.refine_2.refine_2
    ι : Type u_1
    R : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : Finite ι
    O : Type u_4
    inst✝¹ : AddCommGroup O
    inst✝ : Module R O
    M N : Submodule R O
    b'M : Basis ι R (Subtype fun x => Membership.mem M x)
    N_bot : Ne N Bot.bot
    N_le_M : LE.le N M
    this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
    ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
    a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
    a_mem : Membership.mem (ϕ.submoduleImage N) a
    a_zero : Not (Eq a 0)
    y : O
    yN : Membership.mem N y
    ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
    _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
    c : ι → R
    hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
    val✝ : Fintype ι
    y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
    y'M : Membership.mem M y'
    mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
    a_smul_y' : Eq (HSMul.hSMul a y') y
    ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
    ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
    M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
    N' : Submodule R O := Submodule.map N.subtype (LinearMap.ker (ϕ.comp (Submodul …
    M'_le_M : LE.le M' M
    N'_le_M' : LE.le N' M'
    N'_le_N : LE.le N' N
    y'_ortho_M' : ∀ (c : R) (z : O), Membership.mem M' z → Eq (HAdd.hAdd (HSMul.hS …
    ay'_ortho_N' : ∀ (c : R) (z : O), Membership.mem N' z → Eq (HAdd.hAdd (HSMul.h …
    n' : Nat
    bN' : Basis (Fin n') R (Subtype fun x => Membership.mem N' x)
    m' : Nat
    hn'm' : LE.le n' m'
    bM' : Basis (Fin m') R (Subtype fun x => Membership.mem M' x)
    as : Fin n' → R
    h : ∀ (i : Fin n'), Eq (↑(bN' i)) (HSMul.hSMul (as i) ↑(bM' (Fin.castLE hn'm'  …
    i : Fin (HAdd.hAdd n' 1)
    ⊢ Eq (↑((Basis.mkFinConsOfLE y yN bN' N'_le_N ⋯ ⋯) i)) (HSMul.hSMul (Fin.cons  …
  -/
  rw [Basis.coe_mkFinConsOfLE, Basis.coe_mkFinConsOfLE]
  /-
    case neg.intro.intro.intro.refine_2.refine_2
    ι : Type u_1
    R : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : Finite ι
    O : Type u_4
    inst✝¹ : AddCommGroup O
    inst✝ : Module R O
    M N : Submodule R O
    b'M : Basis ι R (Subtype fun x => Membership.mem M x)
    N_bot : Ne N Bot.bot
    N_le_M : LE.le N M
    this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
    ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
    a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
    a_mem : Membership.mem (ϕ.submoduleImage N) a
    a_zero : Not (Eq a 0)
    y : O
    yN : Membership.mem N y
    ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
    _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
    c : ι → R
    hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
    val✝ : Fintype ι
    y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
    y'M : Membership.mem M y'
    mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
    a_smul_y' : Eq (HSMul.hSMul a y') y
    ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
    ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
    M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
    N' : Submodule R O := Submodule.map N.subtype (LinearMap.ker (ϕ.comp (Submodul …
    M'_le_M : LE.le M' M
    N'_le_M' : LE.le N' M'
    N'_le_N : LE.le N' N
    y'_ortho_M' : ∀ (c : R) (z : O), Membership.mem M' z → Eq (HAdd.hAdd (HSMul.hS …
    ay'_ortho_N' : ∀ (c : R) (z : O), Membership.mem N' z → Eq (HAdd.hAdd (HSMul.h …
    n' : Nat
    bN' : Basis (Fin n') R (Subtype fun x => Membership.mem N' x)
    m' : Nat
    hn'm' : LE.le n' m'
    bM' : Basis (Fin m') R (Subtype fun x => Membership.mem M' x)
    as : Fin n' → R
    h : ∀ (i : Fin n'), Eq (↑(bN' i)) (HSMul.hSMul (as i) ↑(bM' (Fin.castLE hn'm'  …
    i : Fin (HAdd.hAdd n' 1)
    ⊢ Eq (↑(Fin.cons ⟨y, yN⟩ (Function.comp ⇑(Submodule.inclusion N'_le_N) ⇑bN') i …
  -/
  refine Fin.cases ?_ (fun i ↦ ?_) i
    /-
      case neg.intro.intro.intro.refine_2.refine_2.refine_1
      ι : Type u_1
      R : Type u_2
      inst✝⁵ : CommRing R
      inst✝⁴ : IsDomain R
      inst✝³ : IsPrincipalIdealRing R
      inst✝² : Finite ι
      O : Type u_4
      inst✝¹ : AddCommGroup O
      inst✝ : Module R O
      M N : Submodule R O
      b'M : Basis ι R (Subtype fun x => Membership.mem M x)
      N_bot : Ne N Bot.bot
      N_le_M : LE.le N M
      this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
      ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
      ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
      a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
      a_mem : Membership.mem (ϕ.submoduleImage N) a
      a_zero : Not (Eq a 0)
      y : O
      yN : Membership.mem N y
      ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
      _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
      c : ι → R
      hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
      val✝ : Fintype ι
      y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
      y'M : Membership.mem M y'
      mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
      a_smul_y' : Eq (HSMul.hSMul a y') y
      ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
      ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
      M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
      N' : Submodule R O := Submodule.map N.subtype (LinearMap.ker (ϕ.comp (Submodul …
      M'_le_M : LE.le M' M
      N'_le_M' : LE.le N' M'
      N'_le_N : LE.le N' N
      y'_ortho_M' : ∀ (c : R) (z : O), Membership.mem M' z → Eq (HAdd.hAdd (HSMul.hS …
      ay'_ortho_N' : ∀ (c : R) (z : O), Membership.mem N' z → Eq (HAdd.hAdd (HSMul.h …
      n' : Nat
      bN' : Basis (Fin n') R (Subtype fun x => Membership.mem N' x)
      m' : Nat
      hn'm' : LE.le n' m'
      bM' : Basis (Fin m') R (Subtype fun x => Membership.mem M' x)
      as : Fin n' → R
      h : ∀ (i : Fin n'), Eq (↑(bN' i)) (HSMul.hSMul (as i) ↑(bM' (Fin.castLE hn'm'  …
      i : Fin (HAdd.hAdd n' 1)
      ⊢ Eq (↑(Fin.cons ⟨y, yN⟩ (Function.comp ⇑(Submodule.inclusion N'_le_N) ⇑bN') 0 …
    -/
  · simp only [Fin.cons_zero, Fin.castLE_zero]
    /-
      case neg.intro.intro.intro.refine_2.refine_2.refine_1
      ι : Type u_1
      R : Type u_2
      inst✝⁵ : CommRing R
      inst✝⁴ : IsDomain R
      inst✝³ : IsPrincipalIdealRing R
      inst✝² : Finite ι
      O : Type u_4
      inst✝¹ : AddCommGroup O
      inst✝ : Module R O
      M N : Submodule R O
      b'M : Basis ι R (Subtype fun x => Membership.mem M x)
      N_bot : Ne N Bot.bot
      N_le_M : LE.le N M
      this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
      ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
      ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
      a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
      a_mem : Membership.mem (ϕ.submoduleImage N) a
      a_zero : Not (Eq a 0)
      y : O
      yN : Membership.mem N y
      ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
      _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
      c : ι → R
      hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
      val✝ : Fintype ι
      y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
      y'M : Membership.mem M y'
      mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
      a_smul_y' : Eq (HSMul.hSMul a y') y
      ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
      ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
      M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
      N' : Submodule R O := Submodule.map N.subtype (LinearMap.ker (ϕ.comp (Submodul …
      M'_le_M : LE.le M' M
      N'_le_M' : LE.le N' M'
      N'_le_N : LE.le N' N
      y'_ortho_M' : ∀ (c : R) (z : O), Membership.mem M' z → Eq (HAdd.hAdd (HSMul.hS …
      ay'_ortho_N' : ∀ (c : R) (z : O), Membership.mem N' z → Eq (HAdd.hAdd (HSMul.h …
      n' : Nat
      bN' : Basis (Fin n') R (Subtype fun x => Membership.mem N' x)
      m' : Nat
      hn'm' : LE.le n' m'
      bM' : Basis (Fin m') R (Subtype fun x => Membership.mem M' x)
      as : Fin n' → R
      h : ∀ (i : Fin n'), Eq (↑(bN' i)) (HSMul.hSMul (as i) ↑(bM' (Fin.castLE hn'm'  …
      i : Fin (HAdd.hAdd n' 1)
      ⊢ Eq y (HSMul.hSMul a y')
    -/
    exact a_smul_y'.symm
    /-
      🎉 no goals
    -/
    /-
      case neg.intro.intro.intro.refine_2.refine_2.refine_2
      ι : Type u_1
      R : Type u_2
      inst✝⁵ : CommRing R
      inst✝⁴ : IsDomain R
      inst✝³ : IsPrincipalIdealRing R
      inst✝² : Finite ι
      O : Type u_4
      inst✝¹ : AddCommGroup O
      inst✝ : Module R O
      M N : Submodule R O
      b'M : Basis ι R (Subtype fun x => Membership.mem M x)
      N_bot : Ne N Bot.bot
      N_le_M : LE.le N M
      this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
      ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
      ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
      a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
      a_mem : Membership.mem (ϕ.submoduleImage N) a
      a_zero : Not (Eq a 0)
      y : O
      yN : Membership.mem N y
      ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
      _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
      c : ι → R
      hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
      val✝ : Fintype ι
      y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
      y'M : Membership.mem M y'
      mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
      a_smul_y' : Eq (HSMul.hSMul a y') y
      ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
      ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
      M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
      N' : Submodule R O := Submodule.map N.subtype (LinearMap.ker (ϕ.comp (Submodul …
      M'_le_M : LE.le M' M
      N'_le_M' : LE.le N' M'
      N'_le_N : LE.le N' N
      y'_ortho_M' : ∀ (c : R) (z : O), Membership.mem M' z → Eq (HAdd.hAdd (HSMul.hS …
      ay'_ortho_N' : ∀ (c : R) (z : O), Membership.mem N' z → Eq (HAdd.hAdd (HSMul.h …
      n' : Nat
      bN' : Basis (Fin n') R (Subtype fun x => Membership.mem N' x)
      m' : Nat
      hn'm' : LE.le n' m'
      bM' : Basis (Fin m') R (Subtype fun x => Membership.mem M' x)
      as : Fin n' → R
      h : ∀ (i : Fin n'), Eq (↑(bN' i)) (HSMul.hSMul (as i) ↑(bM' (Fin.castLE hn'm'  …
      i✝ : Fin (HAdd.hAdd n' 1)
      i : Fin n'
      ⊢ Eq (↑(Fin.cons ⟨y, yN⟩ (Function.comp ⇑(Submodule.inclusion N'_le_N) ⇑bN') i …
    -/
  · rw [Fin.castLE_succ]
    /-
      case neg.intro.intro.intro.refine_2.refine_2.refine_2
      ι : Type u_1
      R : Type u_2
      inst✝⁵ : CommRing R
      inst✝⁴ : IsDomain R
      inst✝³ : IsPrincipalIdealRing R
      inst✝² : Finite ι
      O : Type u_4
      inst✝¹ : AddCommGroup O
      inst✝ : Module R O
      M N : Submodule R O
      b'M : Basis ι R (Subtype fun x => Membership.mem M x)
      N_bot : Ne N Bot.bot
      N_le_M : LE.le N M
      this : Exists fun ϕ => ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membe …
      ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x) R := this.c …
      ϕ_max : ∀ (ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem M x)  …
      a : R := Submodule.IsPrincipal.generator (ϕ.submoduleImage N)
      a_mem : Membership.mem (ϕ.submoduleImage N) a
      a_zero : Not (Eq a 0)
      y : O
      yN : Membership.mem N y
      ϕy_eq : Eq (ϕ ⟨y, ⋯⟩) a
      _ϕy_ne_zero : Ne (ϕ ⟨y, ⋯⟩) 0
      c : ι → R
      hc : ∀ (i : ι), Eq ((b'M.coord i) ⟨y, ⋯⟩) (HMul.hMul a (c i))
      val✝ : Fintype ι
      y' : O := Finset.univ.sum fun i => HSMul.hSMul (c i) ↑(b'M i)
      y'M : Membership.mem M y'
      mk_y' : Eq ⟨y', y'M⟩ (Finset.univ.sum fun i => HSMul.hSMul (c i) (b'M i))
      a_smul_y' : Eq (HSMul.hSMul a y') y
      ϕy'_eq : Eq (ϕ ⟨y', y'M⟩) 1
      ϕy'_ne_zero : Ne (ϕ ⟨y', y'M⟩) 0
      M' : Submodule R O := Submodule.map M.subtype (LinearMap.ker ϕ)
      N' : Submodule R O := Submodule.map N.subtype (LinearMap.ker (ϕ.comp (Submodul …
      M'_le_M : LE.le M' M
      N'_le_M' : LE.le N' M'
      N'_le_N : LE.le N' N
      y'_ortho_M' : ∀ (c : R) (z : O), Membership.mem M' z → Eq (HAdd.hAdd (HSMul.hS …
      ay'_ortho_N' : ∀ (c : R) (z : O), Membership.mem N' z → Eq (HAdd.hAdd (HSMul.h …
      n' : Nat
      bN' : Basis (Fin n') R (Subtype fun x => Membership.mem N' x)
      m' : Nat
      hn'm' : LE.le n' m'
      bM' : Basis (Fin m') R (Subtype fun x => Membership.mem M' x)
      as : Fin n' → R
      h : ∀ (i : Fin n'), Eq (↑(bN' i)) (HSMul.hSMul (as i) ↑(bM' (Fin.castLE hn'm'  …
      i✝ : Fin (HAdd.hAdd n' 1)
      i : Fin n'
      ⊢ Eq (↑(Fin.cons ⟨y, yN⟩ (Function.comp ⇑(Submodule.inclusion N'_le_N) ⇑bN') i …
    -/
    simp only [Fin.cons_succ, Function.comp_apply, coe_inclusion, map_coe, coe_subtype, h i]
    /-
      🎉 no goals
    -/


/-- A submodule of a free `R`-module of finite rank is also a free `R`-module of finite rank,
if `R` is a principal ideal domain.

This is a `lemma` to make the induction a bit easier. To actually access the basis,
see `Submodule.basisOfPid`.

See also the stronger version `Submodule.smithNormalForm`.
-/
theorem Submodule.nonempty_basis_of_pid {ι : Type*} [Finite ι] (b : Basis ι R M)
    (N : Submodule R M) : ∃ n : ℕ, Nonempty (Basis (Fin n) R N) := by
  /-
    R : Type u_2
    inst✝⁵ : CommRing R
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : IsDomain R
    inst✝¹ : IsPrincipalIdealRing R
    ι : Type u_4
    inst✝ : Finite ι
    b : Basis ι R M
    N : Submodule R M
    ⊢ Exists fun n => Nonempty (Basis (Fin n) R (Subtype fun x => Membership.mem N …
  -/
  haveI := Classical.decEq M
  /-
    R : Type u_2
    inst✝⁵ : CommRing R
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : IsDomain R
    inst✝¹ : IsPrincipalIdealRing R
    ι : Type u_4
    inst✝ : Finite ι
    b : Basis ι R M
    N : Submodule R M
    this : DecidableEq M
    ⊢ Exists fun n => Nonempty (Basis (Fin n) R (Subtype fun x => Membership.mem N …
  -/
  cases nonempty_fintype ι
  induction N using inductionOnRank b with | ih N ih =>
  let b' := (b.reindex (Fintype.equivFin ι)).map (LinearEquiv.ofTop _ rfl).symm
  by_cases N_bot : N = ⊥
  · subst N_bot
    exact ⟨0, ⟨Basis.empty _⟩⟩
  obtain ⟨y, -, a, hay, M', -, N', N'_le_N, -, -, ay_ortho, h'⟩ :=
    Submodule.basis_of_pid_aux ⊤ N b' N_bot le_top
  obtain ⟨n', ⟨bN'⟩⟩ := ih N' N'_le_N _ hay ay_ortho
  obtain ⟨bN, _hbN⟩ := h' n' bN'
  exact ⟨n' + 1, ⟨bN⟩⟩


/-- A submodule of a free `R`-module of finite rank is also a free `R`-module of finite rank,
if `R` is a principal ideal domain.

See also the stronger version `Submodule.smithNormalForm`.
-/
noncomputable def Submodule.basisOfPid {ι : Type*} [Finite ι] (b : Basis ι R M)
    (N : Submodule R M) : Σn : ℕ, Basis (Fin n) R N :=
  ⟨_, (N.nonempty_basis_of_pid b).choose_spec.some⟩


theorem Submodule.basisOfPid_bot {ι : Type*} [Finite ι] (b : Basis ι R M) :
    Submodule.basisOfPid b ⊥ = ⟨0, Basis.empty _⟩ := by
  /-
    R : Type u_2
    inst✝⁵ : CommRing R
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : IsDomain R
    inst✝¹ : IsPrincipalIdealRing R
    ι : Type u_4
    inst✝ : Finite ι
    b : Basis ι R M
    ⊢ Eq (Submodule.basisOfPid b Bot.bot) ⟨0, Basis.empty (Subtype fun x => Member …
  -/
  obtain ⟨n, b'⟩ := Submodule.basisOfPid b ⊥
  /-
    case mk
    R : Type u_2
    inst✝⁵ : CommRing R
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : IsDomain R
    inst✝¹ : IsPrincipalIdealRing R
    ι : Type u_4
    inst✝ : Finite ι
    b : Basis ι R M
    n : Nat
    b' : Basis (Fin n) R (Subtype fun x => Membership.mem Bot.bot x)
    ⊢ Eq ⟨n, b'⟩ ⟨0, Basis.empty (Subtype fun x => Membership.mem Bot.bot x)⟩
  -/
  let e : Fin n ≃ Fin 0 := b'.indexEquiv (Basis.empty _ : Basis (Fin 0) R (⊥ : Submodule R M))
  /-
    case mk
    R : Type u_2
    inst✝⁵ : CommRing R
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : IsDomain R
    inst✝¹ : IsPrincipalIdealRing R
    ι : Type u_4
    inst✝ : Finite ι
    b : Basis ι R M
    n : Nat
    b' : Basis (Fin n) R (Subtype fun x => Membership.mem Bot.bot x)
    e : Equiv (Fin n) (Fin 0) := b'.indexEquiv (Basis.empty (Subtype fun x => Memb …
    ⊢ Eq ⟨n, b'⟩ ⟨0, Basis.empty (Subtype fun x => Membership.mem Bot.bot x)⟩
  -/
  obtain rfl : n = 0 := by simpa using Fintype.card_eq.mpr ⟨e⟩
  /-
    case mk
    R : Type u_2
    inst✝⁵ : CommRing R
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : IsDomain R
    inst✝¹ : IsPrincipalIdealRing R
    ι : Type u_4
    inst✝ : Finite ι
    b : Basis ι R M
    b' : Basis (Fin 0) R (Subtype fun x => Membership.mem Bot.bot x)
    e : Equiv (Fin 0) (Fin 0) := b'.indexEquiv (Basis.empty (Subtype fun x => Memb …
    ⊢ Eq ⟨0, b'⟩ ⟨0, Basis.empty (Subtype fun x => Membership.mem Bot.bot x)⟩
  -/
  exact Sigma.eq rfl (Basis.eq_of_apply_eq <| finZeroElim)
  /-
    🎉 no goals
  -/


/-- A submodule inside a free `R`-submodule of finite rank is also a free `R`-module of finite rank,
if `R` is a principal ideal domain.

See also the stronger version `Submodule.smithNormalFormOfLE`.
-/
noncomputable def Submodule.basisOfPidOfLE {ι : Type*} [Finite ι] {N O : Submodule R M}
    (hNO : N ≤ O) (b : Basis ι R O) : Σn : ℕ, Basis (Fin n) R N :=
  let ⟨n, bN'⟩ := Submodule.basisOfPid b (N.comap O.subtype)
  ⟨n, bN'.map (Submodule.comapSubtypeEquivOfLe hNO)⟩


/-- A submodule inside the span of a linear independent family is a free `R`-module of finite rank,
if `R` is a principal ideal domain. -/
noncomputable def Submodule.basisOfPidOfLESpan {ι : Type*} [Finite ι] {b : ι → M}
    (hb : LinearIndependent R b) {N : Submodule R M} (le : N ≤ Submodule.span R (Set.range b)) :
    Σn : ℕ, Basis (Fin n) R N :=
  Submodule.basisOfPidOfLE le (Basis.span hb)


/-- A finite type torsion free module over a PID admits a basis. -/
noncomputable def Module.basisOfFiniteTypeTorsionFree [Fintype ι] {s : ι → M}
    (hs : span R (range s) = ⊤) [NoZeroSMulDivisors R M] : Σn : ℕ, Basis (Fin n) R M := by
  classical
    -- We define `N` as the submodule spanned by a maximal linear independent subfamily of `s`
    have := exists_maximal_independent R s
    let I : Set ι := this.choose
    obtain
      ⟨indepI : LinearIndependent R (s ∘ (fun x => x) : I → M), hI :
        ∀ i ∉ I, ∃ a : R, a ≠ 0 ∧ a • s i ∈ span R (s '' I)⟩ :=
      this.choose_spec
    let N := span R (range <| (s ∘ (fun x => x) : I → M))
    -- same as `span R (s '' I)` but more convenient
    let _sI : I → N := fun i ↦ ⟨s i.1, subset_span (mem_range_self i)⟩
    -- `s` restricted to `I` is a basis of `N`
    let sI_basis : Basis I R N := Basis.span indepI
    -- Our first goal is to build `A ≠ 0` such that `A • M ⊆ N`
    have exists_a : ∀ i : ι, ∃ a : R, a ≠ 0 ∧ a • s i ∈ N := by
      intro i
      by_cases hi : i ∈ I
      · use 1, zero_ne_one.symm
        rw [one_smul]
        exact subset_span (mem_range_self (⟨i, hi⟩ : I))
      · simpa [image_eq_range s I] using hI i hi
    choose a ha ha' using exists_a
    let A := ∏ i, a i
    have hA : A ≠ 0 := by
      rw [Finset.prod_ne_zero_iff]
      simpa using ha
    -- `M ≃ A • M` because `M` is torsion free and `A ≠ 0`
    let φ : M →ₗ[R] M := LinearMap.lsmul R M A
    have : LinearMap.ker φ = ⊥ := @LinearMap.ker_lsmul R M _ _ _ _ _ hA
    let ψ := LinearEquiv.ofInjective φ (LinearMap.ker_eq_bot.mp this)
    have : LinearMap.range φ ≤ N := by
      -- as announced, `A • M ⊆ N`
      suffices ∀ i, φ (s i) ∈ N by
        rw [LinearMap.range_eq_map, ← hs, map_span_le]
        rintro _ ⟨i, rfl⟩
        apply this
      intro i
      calc
        (∏ j ∈ {i}ᶜ, a j) • a i • s i ∈ N := N.smul_mem _ (ha' i)
        _ = (∏ j, a j) • s i := by rw [Fintype.prod_eq_prod_compl_mul i, mul_smul]

    -- Since a submodule of a free `R`-module is free, we get that `A • M` is free
    obtain ⟨n, b : Basis (Fin n) R (LinearMap.range φ)⟩ := Submodule.basisOfPidOfLE this sI_basis
    -- hence `M` is free.
    exact ⟨n, b.map ψ.symm⟩


theorem Module.free_of_finite_type_torsion_free [_root_.Finite ι] {s : ι → M}
    (hs : span R (range s) = ⊤) [NoZeroSMulDivisors R M] : Module.Free R M := by
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁶ : CommRing R
    M : Type u_3
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : IsDomain R
    inst✝² : IsPrincipalIdealRing R
    inst✝¹ : Finite ι
    s : ι → M
    hs : Eq (Submodule.span R (Set.range s)) Top.top
    inst✝ : NoZeroSMulDivisors R M
    ⊢ Module.Free R M
  -/
  cases nonempty_fintype ι
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    inst✝⁶ : CommRing R
    M : Type u_3
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : IsDomain R
    inst✝² : IsPrincipalIdealRing R
    inst✝¹ : Finite ι
    s : ι → M
    hs : Eq (Submodule.span R (Set.range s)) Top.top
    inst✝ : NoZeroSMulDivisors R M
    val✝ : Fintype ι
    ⊢ Module.Free R M
  -/
  obtain ⟨n, b⟩ : Σn, Basis (Fin n) R M := Module.basisOfFiniteTypeTorsionFree hs
  /-
    case intro.mk
    ι : Type u_1
    R : Type u_2
    inst✝⁶ : CommRing R
    M : Type u_3
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : IsDomain R
    inst✝² : IsPrincipalIdealRing R
    inst✝¹ : Finite ι
    s : ι → M
    hs : Eq (Submodule.span R (Set.range s)) Top.top
    inst✝ : NoZeroSMulDivisors R M
    val✝ : Fintype ι
    n : Nat
    b : Basis (Fin n) R M
    ⊢ Module.Free R M
  -/
  exact Module.Free.of_basis b
  /-
    🎉 no goals
  -/


/-- A finite type torsion free module over a PID admits a basis. -/
noncomputable def Module.basisOfFiniteTypeTorsionFree' [Module.Finite R M]
    [NoZeroSMulDivisors R M] : Σn : ℕ, Basis (Fin n) R M :=
  Module.basisOfFiniteTypeTorsionFree Module.Finite.exists_fin.choose_spec.choose_spec


instance Module.free_of_finite_type_torsion_free' [Module.Finite R M] [NoZeroSMulDivisors R M] :
    Module.Free R M := by
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁶ : CommRing R
    M : Type u_3
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    b : ι → M
    inst✝³ : IsDomain R
    inst✝² : IsPrincipalIdealRing R
    inst✝¹ : Module.Finite R M
    inst✝ : NoZeroSMulDivisors R M
    ⊢ Module.Free R M
  -/
  obtain ⟨n, b⟩ : Σn, Basis (Fin n) R M := Module.basisOfFiniteTypeTorsionFree'
  /-
    case mk
    ι : Type u_1
    R : Type u_2
    inst✝⁶ : CommRing R
    M : Type u_3
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    b✝ : ι → M
    inst✝³ : IsDomain R
    inst✝² : IsPrincipalIdealRing R
    inst✝¹ : Module.Finite R M
    inst✝ : NoZeroSMulDivisors R M
    n : Nat
    b : Basis (Fin n) R M
    ⊢ Module.Free R M
  -/
  exact Module.Free.of_basis b
  /-
    🎉 no goals
  -/


instance {S : Type*} [CommRing S] [Algebra R S] {I : Ideal S} [hI₁ : Module.Finite R I]
    [hI₂ : NoZeroSMulDivisors R I] : Module.Free R I := by
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁶ : CommRing R
    M : Type u_3
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    b : ι → M
    inst✝³ : IsDomain R
    inst✝² : IsPrincipalIdealRing R
    S : Type u_4
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    I : Ideal S
    hI₁ : Module.Finite R (Subtype fun x => Membership.mem I x)
    hI₂ : NoZeroSMulDivisors R (Subtype fun x => Membership.mem I x)
    ⊢ Module.Free R (Subtype fun x => Membership.mem I x)
  -/
  have : Module.Finite R (restrictScalars R I) := hI₁
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁶ : CommRing R
    M : Type u_3
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    b : ι → M
    inst✝³ : IsDomain R
    inst✝² : IsPrincipalIdealRing R
    S : Type u_4
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    I : Ideal S
    hI₁ : Module.Finite R (Subtype fun x => Membership.mem I x)
    hI₂ : NoZeroSMulDivisors R (Subtype fun x => Membership.mem I x)
    this : Module.Finite R (Subtype fun x => Membership.mem (Submodule.restrictSca …
    ⊢ Module.Free R (Subtype fun x => Membership.mem I x)
  -/
  have : NoZeroSMulDivisors R (restrictScalars R I) := hI₂
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁶ : CommRing R
    M : Type u_3
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    b : ι → M
    inst✝³ : IsDomain R
    inst✝² : IsPrincipalIdealRing R
    S : Type u_4
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    I : Ideal S
    hI₁ : Module.Finite R (Subtype fun x => Membership.mem I x)
    hI₂ : NoZeroSMulDivisors R (Subtype fun x => Membership.mem I x)
    this✝ : Module.Finite R (Subtype fun x => Membership.mem (Submodule.restrictSc …
    this : NoZeroSMulDivisors R (Subtype fun x => Membership.mem (Submodule.restri …
    ⊢ Module.Free R (Subtype fun x => Membership.mem I x)
  -/
  change Module.Free R (restrictScalars R I)
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁶ : CommRing R
    M : Type u_3
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    b : ι → M
    inst✝³ : IsDomain R
    inst✝² : IsPrincipalIdealRing R
    S : Type u_4
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    I : Ideal S
    hI₁ : Module.Finite R (Subtype fun x => Membership.mem I x)
    hI₂ : NoZeroSMulDivisors R (Subtype fun x => Membership.mem I x)
    this✝ : Module.Finite R (Subtype fun x => Membership.mem (Submodule.restrictSc …
    this : NoZeroSMulDivisors R (Subtype fun x => Membership.mem (Submodule.restri …
    ⊢ Module.Free R (Subtype fun x => Membership.mem (Submodule.restrictScalars R  …
  -/
  exact Module.free_of_finite_type_torsion_free'
  /-
    🎉 no goals
  -/


theorem Module.free_iff_noZeroSMulDivisors [Module.Finite R M] :
    Module.Free R M ↔ NoZeroSMulDivisors R M :=
  ⟨fun _ ↦ inferInstance, fun _ ↦ inferInstance⟩


/-- A Smith normal form basis for a submodule `N` of a module `M` consists of
bases for `M` and `N` such that the inclusion map `N → M` can be written as a
(rectangular) matrix with `a` along the diagonal: in Smith normal form. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): @[nolint has_nonempty_instance]
structure Basis.SmithNormalForm (N : Submodule R M) (ι : Type*) (n : ℕ) where
  /-- The basis of M. -/
  bM : Basis ι R M
  /-- The basis of N. -/
  bN : Basis (Fin n) R N
  /-- The mapping between the vectors of the bases. -/
  f : Fin n ↪ ι
  /-- The (diagonal) entries of the matrix. -/
  a : Fin n → R
  /-- The SNF relation between the vectors of the bases. -/
  snf : ∀ i, (bN i : M) = a i • bM (f i)


lemma repr_eq_zero_of_nmem_range {i : ι} (hi : i ∉ Set.range snf.f) :
    snf.bM.repr m i = 0 := by
  /-
    ι : Type u_1
    R : Type u_2
    inst✝² : CommRing R
    M : Type u_3
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    N : Submodule R M
    snf : Basis.SmithNormalForm N ι n
    m : Subtype fun x => Membership.mem N x
    i : ι
    hi : Not (Membership.mem (Set.range ⇑snf.f) i)
    ⊢ Eq ((snf.bM.repr ↑m) i) 0
  -/
  obtain ⟨m, hm⟩ := m
  /-
    case mk
    ι : Type u_1
    R : Type u_2
    inst✝² : CommRing R
    M : Type u_3
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    N : Submodule R M
    snf : Basis.SmithNormalForm N ι n
    i : ι
    hi : Not (Membership.mem (Set.range ⇑snf.f) i)
    m : M
    hm : Membership.mem N m
    ⊢ Eq ((snf.bM.repr ↑⟨m, hm⟩) i) 0
  -/
  obtain ⟨c, rfl⟩ := snf.bN.mem_submodule_iff.mp hm
  /-
    case mk.intro
    ι : Type u_1
    R : Type u_2
    inst✝² : CommRing R
    M : Type u_3
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    N : Submodule R M
    snf : Basis.SmithNormalForm N ι n
    i : ι
    hi : Not (Membership.mem (Set.range ⇑snf.f) i)
    c : Finsupp (Fin n) R
    hm : Membership.mem N (c.sum fun i x => HSMul.hSMul x ↑(snf.bN i))
    ⊢ Eq ((snf.bM.repr ↑⟨c.sum fun i x => HSMul.hSMul x ↑(snf.bN i), hm⟩) i) 0
  -/
  replace hi : ∀ j, snf.f j ≠ i := by simpa using hi
  /-
    case mk.intro
    ι : Type u_1
    R : Type u_2
    inst✝² : CommRing R
    M : Type u_3
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    N : Submodule R M
    snf : Basis.SmithNormalForm N ι n
    i : ι
    c : Finsupp (Fin n) R
    hm : Membership.mem N (c.sum fun i x => HSMul.hSMul x ↑(snf.bN i))
    hi : ∀ (j : Fin n), Ne (snf.f j) i
    ⊢ Eq ((snf.bM.repr ↑⟨c.sum fun i x => HSMul.hSMul x ↑(snf.bN i), hm⟩) i) 0
  -/
  simp [Finsupp.single_apply, hi, snf.snf, map_finsupp_sum]
  /-
    🎉 no goals
  -/


lemma le_ker_coord_of_nmem_range {i : ι} (hi : i ∉ Set.range snf.f) :
    N ≤ LinearMap.ker (snf.bM.coord i) :=
  fun m hm ↦ snf.repr_eq_zero_of_nmem_range ⟨m, hm⟩ hi


@[simp] lemma repr_apply_embedding_eq_repr_smul {i : Fin n} :
    snf.bM.repr m (snf.f i) = snf.bN.repr (snf.a i • m) i := by
  /-
    ι : Type u_1
    R : Type u_2
    inst✝² : CommRing R
    M : Type u_3
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    N : Submodule R M
    snf : Basis.SmithNormalForm N ι n
    m : Subtype fun x => Membership.mem N x
    i : Fin n
    ⊢ Eq ((snf.bM.repr ↑m) (snf.f i)) ((snf.bN.repr (HSMul.hSMul (snf.a i) m)) i)
  -/
  obtain ⟨m, hm⟩ := m
  /-
    case mk
    ι : Type u_1
    R : Type u_2
    inst✝² : CommRing R
    M : Type u_3
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    N : Submodule R M
    snf : Basis.SmithNormalForm N ι n
    i : Fin n
    m : M
    hm : Membership.mem N m
    ⊢ Eq ((snf.bM.repr ↑⟨m, hm⟩) (snf.f i)) ((snf.bN.repr (HSMul.hSMul (snf.a i) ⟨ …
  -/
  obtain ⟨c, rfl⟩ := snf.bN.mem_submodule_iff.mp hm
  replace hm : (⟨Finsupp.sum c fun i t ↦ t • (↑(snf.bN i) : M), hm⟩ : N) =
      Finsupp.sum c fun i t ↦ t • ⟨snf.bN i, (snf.bN i).2⟩ := by
    ext; change _ = N.subtype _; simp [map_finsupp_sum]
  classical
  simp_rw [hm, map_smul, map_finsupp_sum, map_smul, Subtype.coe_eta, repr_self,
    Finsupp.smul_single, smul_eq_mul, mul_one, Finsupp.sum_single, Finsupp.smul_apply, snf.snf,
    map_smul, repr_self, Finsupp.smul_single, smul_eq_mul, mul_one, Finsupp.sum_apply,
    Finsupp.single_apply, EmbeddingLike.apply_eq_iff_eq, Finsupp.sum_ite_eq',
    Finsupp.mem_support_iff, ite_not, mul_comm, ite_eq_right_iff]
  exact fun a ↦ (mul_eq_zero_of_right _ a).symm


@[simp] lemma repr_comp_embedding_eq_smul :
    snf.bM.repr m ∘ snf.f = snf.a • (snf.bN.repr m : Fin n → R) := by
  /-
    ι : Type u_1
    R : Type u_2
    inst✝² : CommRing R
    M : Type u_3
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    N : Submodule R M
    snf : Basis.SmithNormalForm N ι n
    m : Subtype fun x => Membership.mem N x
    ⊢ Eq (Function.comp ⇑(snf.bM.repr ↑m) ⇑snf.f) (HSMul.hSMul snf.a ⇑(snf.bN.repr …
  -/
  ext i
  /-
    case h
    ι : Type u_1
    R : Type u_2
    inst✝² : CommRing R
    M : Type u_3
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    N : Submodule R M
    snf : Basis.SmithNormalForm N ι n
    m : Subtype fun x => Membership.mem N x
    i : Fin n
    ⊢ Eq (Function.comp (⇑(snf.bM.repr ↑m)) (⇑snf.f) i) (HSMul.hSMul snf.a (⇑(snf. …
  -/
  simp [Pi.smul_apply (snf.a i)]
  /-
    🎉 no goals
  -/


@[simp] lemma coord_apply_embedding_eq_smul_coord {i : Fin n} :
    snf.bM.coord (snf.f i) ∘ₗ N.subtype = snf.a i • snf.bN.coord i := by
  /-
    ι : Type u_1
    R : Type u_2
    inst✝² : CommRing R
    M : Type u_3
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    N : Submodule R M
    snf : Basis.SmithNormalForm N ι n
    i : Fin n
    ⊢ Eq ((snf.bM.coord (snf.f i)).comp N.subtype) (HSMul.hSMul (snf.a i) (snf.bN. …
  -/
  ext m
  /-
    case h
    ι : Type u_1
    R : Type u_2
    inst✝² : CommRing R
    M : Type u_3
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    N : Submodule R M
    snf : Basis.SmithNormalForm N ι n
    i : Fin n
    m : Subtype fun x => Membership.mem N x
    ⊢ Eq (((snf.bM.coord (snf.f i)).comp N.subtype) m) ((HSMul.hSMul (snf.a i) (sn …
  -/
  simp [Pi.smul_apply (snf.a i)]
  /-
    🎉 no goals
  -/


/-- Given a Smith-normal-form pair of bases for `N ⊆ M`, and a linear endomorphism `f` of `M`
that preserves `N`, the diagonal of the matrix of the restriction `f` to `N` does not depend on
which of the two bases for `N` is used. -/
@[simp]
lemma toMatrix_restrict_eq_toMatrix [Fintype ι] [DecidableEq ι]
    (f : M →ₗ[R] M) (hf : ∀ x, f x ∈ N) (hf' : ∀ x ∈ N, f x ∈ N := fun x _ ↦ hf x) {i : Fin n} :
    LinearMap.toMatrix snf.bN snf.bN (LinearMap.restrict f hf') i i =
    LinearMap.toMatrix snf.bM snf.bM f (snf.f i) (snf.f i) := by
  rw [LinearMap.toMatrix_apply, LinearMap.toMatrix_apply,
    snf.repr_apply_embedding_eq_repr_smul ⟨_, (hf _)⟩]
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁴ : CommRing R
    M : Type u_3
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    n : Nat
    N : Submodule R M
    snf : Basis.SmithNormalForm N ι n
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    f : LinearMap (RingHom.id R) M M
    hf : ∀ (x : M), Membership.mem N (f x)
    hf' : optParam (∀ (x : M), Membership.mem N x → Membership.mem N (f x)) ⋯
    i : Fin n
    ⊢ Eq ((snf.bN.repr ((f.restrict hf') (snf.bN i))) i) ((snf.bN.repr (HSMul.hSMu …
  -/
  congr
  /-
    case e_a.h.e_6.h
    ι : Type u_1
    R : Type u_2
    inst✝⁴ : CommRing R
    M : Type u_3
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    n : Nat
    N : Submodule R M
    snf : Basis.SmithNormalForm N ι n
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    f : LinearMap (RingHom.id R) M M
    hf : ∀ (x : M), Membership.mem N (f x)
    hf' : optParam (∀ (x : M), Membership.mem N x → Membership.mem N (f x)) ⋯
    i : Fin n
    ⊢ Eq ((f.restrict hf') (snf.bN i)) (HSMul.hSMul (snf.a i) ⟨f (snf.bM (snf.f i) …
  -/
  ext
  /-
    case e_a.h.e_6.h.a
    ι : Type u_1
    R : Type u_2
    inst✝⁴ : CommRing R
    M : Type u_3
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    n : Nat
    N : Submodule R M
    snf : Basis.SmithNormalForm N ι n
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    f : LinearMap (RingHom.id R) M M
    hf : ∀ (x : M), Membership.mem N (f x)
    hf' : optParam (∀ (x : M), Membership.mem N x → Membership.mem N (f x)) ⋯
    i : Fin n
    ⊢ Eq ↑((f.restrict hf') (snf.bN i)) ↑(HSMul.hSMul (snf.a i) ⟨f (snf.bM (snf.f  …
  -/
  simp [snf.snf]
  /-
    🎉 no goals
  -/


/-- If `M` is finite free over a PID `R`, then any submodule `N` is free
and we can find a basis for `M` and `N` such that the inclusion map is a diagonal matrix
in Smith normal form.

See `Submodule.smithNormalFormOfLE` for a version of this theorem that returns
a `Basis.SmithNormalForm`.

This is a strengthening of `Submodule.basisOfPidOfLE`.
-/
theorem Submodule.exists_smith_normal_form_of_le [Finite ι] (b : Basis ι R M) (N O : Submodule R M)
    (N_le_O : N ≤ O) :
    ∃ (n o : ℕ) (hno : n ≤ o) (bO : Basis (Fin o) R O) (bN : Basis (Fin n) R N) (a : Fin n → R),
      ∀ i, (bN i : M) = a i • bO (Fin.castLE hno i) := by
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁵ : CommRing R
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : IsDomain R
    inst✝¹ : IsPrincipalIdealRing R
    inst✝ : Finite ι
    b : Basis ι R M
    N O : Submodule R M
    N_le_O : LE.le N O
    ⊢ Exists fun n => Exists fun o => Exists fun hno => Exists fun bO => Exists fu …
  -/
  cases nonempty_fintype ι
  induction O using inductionOnRank b generalizing N with | ih M0 ih =>
  obtain ⟨m, b'M⟩ := M0.basisOfPid b
  by_cases N_bot : N = ⊥
  · subst N_bot
    exact ⟨0, m, Nat.zero_le _, b'M, Basis.empty _, finZeroElim, finZeroElim⟩
  obtain ⟨y, hy, a, _, M', M'_le_M, N', _, N'_le_M', y_ortho, _, h⟩ :=
    Submodule.basis_of_pid_aux M0 N b'M N_bot N_le_O

  obtain ⟨n', m', hn'm', bM', bN', as', has'⟩ := ih M' M'_le_M y hy y_ortho N' N'_le_M'
  obtain ⟨bN, h'⟩ := h n' bN'
  obtain ⟨hmn, bM, h''⟩ := h' m' hn'm' bM'
  obtain ⟨as, has⟩ := h'' as' has'
  exact ⟨_, _, hmn, bM, bN, as, has⟩


/-- If `M` is finite free over a PID `R`, then any submodule `N` is free
and we can find a basis for `M` and `N` such that the inclusion map is a diagonal matrix
in Smith normal form.

See `Submodule.exists_smith_normal_form_of_le` for a version of this theorem that doesn't
need to map `N` into a submodule of `O`.

This is a strengthening of `Submodule.basisOfPidOfLe`.
-/
noncomputable def Submodule.smithNormalFormOfLE [Finite ι] (b : Basis ι R M) (N O : Submodule R M)
    (N_le_O : N ≤ O) : Σo n : ℕ, Basis.SmithNormalForm (N.comap O.subtype) (Fin o) n := by
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁵ : CommRing R
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    b✝ : ι → M
    inst✝² : IsDomain R
    inst✝¹ : IsPrincipalIdealRing R
    inst✝ : Finite ι
    b : Basis ι R M
    N O : Submodule R M
    N_le_O : LE.le N O
    ⊢ Sigma fun o => Sigma fun n => Basis.SmithNormalForm (Submodule.comap O.subty …
  -/
  choose n o hno bO bN a snf using N.exists_smith_normal_form_of_le b O N_le_O
  refine
    ⟨o, n, bO, bN.map (comapSubtypeEquivOfLe N_le_O).symm, (Fin.castLEEmb hno), a,
      fun i ↦ ?_⟩
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁵ : CommRing R
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    b✝ : ι → M
    inst✝² : IsDomain R
    inst✝¹ : IsPrincipalIdealRing R
    inst✝ : Finite ι
    b : Basis ι R M
    N O : Submodule R M
    N_le_O : LE.le N O
    n o : Nat
    hno : LE.le n o
    bO : Basis (Fin o) R (Subtype fun x => Membership.mem O x)
    bN : Basis (Fin n) R (Subtype fun x => Membership.mem N x)
    a : Fin n → R
    snf : ∀ (i : Fin n), Eq (↑(bN i)) (HSMul.hSMul (a i) ↑(bO (Fin.castLE hno i)))
    i : Fin n
    ⊢ Eq (↑((bN.map (Submodule.comapSubtypeEquivOfLe N_le_O).symm) i)) (HSMul.hSMu …
  -/
  ext
  simp only [snf, Basis.map_apply, Submodule.comapSubtypeEquivOfLe_symm_apply,
    Submodule.coe_smul_of_tower, Fin.castLEEmb_apply]


/-- If `M` is finite free over a PID `R`, then any submodule `N` is free
and we can find a basis for `M` and `N` such that the inclusion map is a diagonal matrix
in Smith normal form.

This is a strengthening of `Submodule.basisOfPid`.

See also `Ideal.smithNormalForm`, which moreover proves that the dimension of
an ideal is the same as the dimension of the whole ring.
-/
noncomputable def Submodule.smithNormalForm [Finite ι] (b : Basis ι R M) (N : Submodule R M) :
    Σn : ℕ, Basis.SmithNormalForm N ι n :=
  let ⟨m, n, bM, bN, f, a, snf⟩ := N.smithNormalFormOfLE b ⊤ le_top
  let bM' := bM.map (LinearEquiv.ofTop _ rfl)
  let e := bM'.indexEquiv b
  ⟨n, bM'.reindex e, bN.map (comapSubtypeEquivOfLe le_top), f.trans e.toEmbedding, a, fun i ↦ by
    simp only [bM', snf, Basis.map_apply, LinearEquiv.ofTop_apply, Submodule.coe_smul_of_tower,
      Submodule.comapSubtypeEquivOfLe_apply_coe, Basis.reindex_apply,
      Equiv.toEmbedding_apply, Function.Embedding.trans_apply, Equiv.symm_apply_apply]⟩


/-- If `S` a finite-dimensional ring extension of a PID `R` which is free as an `R`-module,
then any nonzero `S`-ideal `I` is free as an `R`-submodule of `S`, and we can
find a basis for `S` and `I` such that the inclusion map is a square diagonal
matrix.

See `Ideal.exists_smith_normal_form` for a version of this theorem that doesn't
need to map `I` into a submodule of `R`.

This is a strengthening of `Submodule.basisOfPid`.
-/
noncomputable def Ideal.smithNormalForm [Fintype ι] (b : Basis ι R S) (I : Ideal S) (hI : I ≠ ⊥) :
    Basis.SmithNormalForm (I.restrictScalars R) ι (Fintype.card ι) :=
  let ⟨n, bS, bI, f, a, snf⟩ := (I.restrictScalars R).smithNormalForm b
  have eq := Ideal.rank_eq bS hI (bI.map ((restrictScalarsEquiv R S S I).restrictScalars R))
                                                                    /-
                                                                      ι : Type u_1
                                                                      R : Type u_2
                                                                      inst✝⁸ : CommRing R
                                                                      M : Type u_3
                                                                      inst✝⁷ : AddCommGroup M
                                                                      inst✝⁶ : Module R M
                                                                      b✝ : ι → M
                                                                      inst✝⁵ : IsDomain R
                                                                      inst✝⁴ : IsPrincipalIdealRing R
                                                                      S : Type u_4
                                                                      inst✝³ : CommRing S
                                                                      inst✝² : IsDomain S
                                                                      inst✝¹ : Algebra R S
                                                                      inst✝ : Fintype ι
                                                                      b : Basis ι R S
                                                                      I : Ideal S
                                                                      hI : Ne I Bot.bot
                                                                      n : Nat
                                                                      bS : Basis ι R S
                                                                      bI : Basis (Fin n) R (Subtype fun x => Membership.mem (Submodule.restrictScala …
                                                                      f : Function.Embedding (Fin n) ι
                                                                      a : Fin n → R
                                                                      snf : ∀ (i : Fin n), Eq (↑(bI i)) (HSMul.hSMul (a i) (bS (f i)))
                                                                      eq : Eq (Fintype.card (Fin n)) (Fintype.card ι)
                                                                      ⊢ Eq (Fintype.card (Fin n)) (Fintype.card (Fin (Fintype.card ι)))
                                                                    -/
  let e : Fin n ≃ Fin (Fintype.card ι) := Fintype.equivOfCardEq (by rw [eq, Fintype.card_fin])
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  ⟨bS, bI.reindex e, e.symm.toEmbedding.trans f, a ∘ e.symm, fun i ↦ by
    simp only [snf, Basis.coe_reindex, Function.Embedding.trans_apply, Equiv.toEmbedding_apply,
      (· ∘ ·)]⟩


/-- If `S` a finite-dimensional ring extension of a PID `R` which is free as an `R`-module,
then any nonzero `S`-ideal `I` is free as an `R`-submodule of `S`, and we can
find a basis for `S` and `I` such that the inclusion map is a square diagonal
matrix.

See also `Ideal.smithNormalForm` for a version of this theorem that returns
a `Basis.SmithNormalForm`.

The definitions `Ideal.ringBasis`, `Ideal.selfBasis`, `Ideal.smithCoeffs` are (noncomputable)
choices of values for this existential quantifier.
-/
theorem Ideal.exists_smith_normal_form (b : Basis ι R S) (I : Ideal S) (hI : I ≠ ⊥) :
    ∃ (b' : Basis ι R S) (a : ι → R) (ab' : Basis ι R I), ∀ i, (ab' i : S) = a i • b' i := by
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : IsDomain R
    inst✝⁴ : IsPrincipalIdealRing R
    S : Type u_4
    inst✝³ : CommRing S
    inst✝² : IsDomain S
    inst✝¹ : Algebra R S
    inst✝ : Finite ι
    b : Basis ι R S
    I : Ideal S
    hI : Ne I Bot.bot
    ⊢ Exists fun b' => Exists fun a => Exists fun ab' => ∀ (i : ι), Eq (↑(ab' i))  …
  -/
  cases nonempty_fintype ι
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : IsDomain R
    inst✝⁴ : IsPrincipalIdealRing R
    S : Type u_4
    inst✝³ : CommRing S
    inst✝² : IsDomain S
    inst✝¹ : Algebra R S
    inst✝ : Finite ι
    b : Basis ι R S
    I : Ideal S
    hI : Ne I Bot.bot
    val✝ : Fintype ι
    ⊢ Exists fun b' => Exists fun a => Exists fun ab' => ∀ (i : ι), Eq (↑(ab' i))  …
  -/
  let ⟨bS, bI, f, a, snf⟩ := I.smithNormalForm b hI
  let e : Fin (Fintype.card ι) ≃ ι :=
    Equiv.ofBijective f
      ((Fintype.bijective_iff_injective_and_card f).mpr ⟨f.injective, Fintype.card_fin _⟩)
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : IsDomain R
    inst✝⁴ : IsPrincipalIdealRing R
    S : Type u_4
    inst✝³ : CommRing S
    inst✝² : IsDomain S
    inst✝¹ : Algebra R S
    inst✝ : Finite ι
    b : Basis ι R S
    I : Ideal S
    hI : Ne I Bot.bot
    val✝ : Fintype ι
    bS : Basis ι R S
    bI : Basis (Fin (Fintype.card ι)) R (Subtype fun x => Membership.mem (Submodul …
    f : Function.Embedding (Fin (Fintype.card ι)) ι
    a : Fin (Fintype.card ι) → R
    snf : ∀ (i : Fin (Fintype.card ι)), Eq (↑(bI i)) (HSMul.hSMul (a i) (bS (f i)))
    e : Equiv (Fin (Fintype.card ι)) ι := Equiv.ofBijective ⇑f ⋯
    ⊢ Exists fun b' => Exists fun a => Exists fun ab' => ∀ (i : ι), Eq (↑(ab' i))  …
  -/
  have fe : ∀ i, f (e.symm i) = i := e.apply_symm_apply
  exact
    ⟨bS, a ∘ e.symm, (bI.reindex e).map ((restrictScalarsEquiv R S _ _).restrictScalars R),
      fun i ↦ by
        simp only [snf, fe, Basis.map_apply, LinearEquiv.restrictScalars_apply R,
          Submodule.restrictScalarsEquiv_apply, Basis.coe_reindex, (· ∘ ·)]⟩


/-- If `S` a finite-dimensional ring extension of a PID `R` which is free as an `R`-module,
then any nonzero `S`-ideal `I` is free as an `R`-submodule of `S`, and we can
find a basis for `S` and `I` such that the inclusion map is a square diagonal
matrix; this is the basis for `S`.
See `Ideal.selfBasis` for the basis on `I`,
see `Ideal.smithCoeffs` for the entries of the diagonal matrix
and `Ideal.selfBasis_def` for the proof that the inclusion map forms a square diagonal matrix.
-/
noncomputable def Ideal.ringBasis (b : Basis ι R S) (I : Ideal S) (hI : I ≠ ⊥) : Basis ι R S :=
  (Ideal.exists_smith_normal_form b I hI).choose


/-- If `S` a finite-dimensional ring extension of a PID `R` which is free as an `R`-module,
then any nonzero `S`-ideal `I` is free as an `R`-submodule of `S`, and we can
find a basis for `S` and `I` such that the inclusion map is a square diagonal
matrix; this is the basis for `I`.
See `Ideal.ringBasis` for the basis on `S`,
see `Ideal.smithCoeffs` for the entries of the diagonal matrix
and `Ideal.selfBasis_def` for the proof that the inclusion map forms a square diagonal matrix.
-/
noncomputable def Ideal.selfBasis (b : Basis ι R S) (I : Ideal S) (hI : I ≠ ⊥) : Basis ι R I :=
  (Ideal.exists_smith_normal_form b I hI).choose_spec.choose_spec.choose


/-- If `S` a finite-dimensional ring extension of a PID `R` which is free as an `R`-module,
then any nonzero `S`-ideal `I` is free as an `R`-submodule of `S`, and we can
find a basis for `S` and `I` such that the inclusion map is a square diagonal
matrix; these are the entries of the diagonal matrix.
See `Ideal.ringBasis` for the basis on `S`,
see `Ideal.selfBasis` for the basis on `I`,
and `Ideal.selfBasis_def` for the proof that the inclusion map forms a square diagonal matrix.
-/
noncomputable def Ideal.smithCoeffs (b : Basis ι R S) (I : Ideal S) (hI : I ≠ ⊥) : ι → R :=
  (Ideal.exists_smith_normal_form b I hI).choose_spec.choose


/-- If `S` a finite-dimensional ring extension of a PID `R` which is free as an `R`-module,
then any nonzero `S`-ideal `I` is free as an `R`-submodule of `S`, and we can
find a basis for `S` and `I` such that the inclusion map is a square diagonal
matrix.
-/
@[simp]
theorem Ideal.selfBasis_def (b : Basis ι R S) (I : Ideal S) (hI : I ≠ ⊥) :
    ∀ i, (Ideal.selfBasis b I hI i : S) = Ideal.smithCoeffs b I hI i • Ideal.ringBasis b I hI i :=
  (Ideal.exists_smith_normal_form b I hI).choose_spec.choose_spec.choose_spec


@[simp]
theorem Ideal.smithCoeffs_ne_zero (b : Basis ι R S) (I : Ideal S) (hI : I ≠ ⊥) (i) :
    Ideal.smithCoeffs b I hI i ≠ 0 := by
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : IsDomain R
    inst✝⁴ : IsPrincipalIdealRing R
    S : Type u_4
    inst✝³ : CommRing S
    inst✝² : IsDomain S
    inst✝¹ : Algebra R S
    inst✝ : Finite ι
    b : Basis ι R S
    I : Ideal S
    hI : Ne I Bot.bot
    i : ι
    ⊢ Ne (Ideal.smithCoeffs b I hI i) 0
  -/
  intro hi
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : IsDomain R
    inst✝⁴ : IsPrincipalIdealRing R
    S : Type u_4
    inst✝³ : CommRing S
    inst✝² : IsDomain S
    inst✝¹ : Algebra R S
    inst✝ : Finite ι
    b : Basis ι R S
    I : Ideal S
    hI : Ne I Bot.bot
    i : ι
    hi : Eq (Ideal.smithCoeffs b I hI i) 0
    ⊢ False
  -/
  apply Basis.ne_zero (Ideal.selfBasis b I hI) i
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : IsDomain R
    inst✝⁴ : IsPrincipalIdealRing R
    S : Type u_4
    inst✝³ : CommRing S
    inst✝² : IsDomain S
    inst✝¹ : Algebra R S
    inst✝ : Finite ι
    b : Basis ι R S
    I : Ideal S
    hI : Ne I Bot.bot
    i : ι
    hi : Eq (Ideal.smithCoeffs b I hI i) 0
    ⊢ Eq ((Ideal.selfBasis b I hI) i) 0
  -/
  refine Subtype.coe_injective ?_
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : IsDomain R
    inst✝⁴ : IsPrincipalIdealRing R
    S : Type u_4
    inst✝³ : CommRing S
    inst✝² : IsDomain S
    inst✝¹ : Algebra R S
    inst✝ : Finite ι
    b : Basis ι R S
    I : Ideal S
    hI : Ne I Bot.bot
    i : ι
    hi : Eq (Ideal.smithCoeffs b I hI i) 0
    ⊢ Eq ((fun a => ↑a) ((Ideal.selfBasis b I hI) i)) ((fun a => ↑a) 0)
  -/
  simp [hi]
  /-
    🎉 no goals
  -/

-- Porting note: can be inferred in Lean 4 so no longer necessary


/-- A set of linearly independent vectors in a module `M` over a semiring `S` is also linearly
independent over a subring `R` of `K`. -/
theorem LinearIndependent.restrict_scalars_algebras {R S M ι : Type*} [CommSemiring R] [Semiring S]
    [AddCommMonoid M] [Algebra R S] [Module R M] [Module S M] [IsScalarTower R S M]
    (hinj : Function.Injective (algebraMap R S)) {v : ι → M} (li : LinearIndependent S v) :
    LinearIndependent R v :=
                                         /-
                                           R : Type u_1
                                           S : Type u_2
                                           M : Type u_3
                                           ι : Type u_4
                                           inst✝⁶ : CommSemiring R
                                           inst✝⁵ : Semiring S
                                           inst✝⁴ : AddCommMonoid M
                                           inst✝³ : Algebra R S
                                           inst✝² : Module R M
                                           inst✝¹ : Module S M
                                           inst✝ : IsScalarTower R S M
                                           hinj : Function.Injective ⇑(algebraMap R S)
                                           v : ι → M
                                           li : LinearIndependent S v
                                           ⊢ Function.Injective fun r => HSMul.hSMul r 1
                                         -/
  LinearIndependent.restrict_scalars (by rwa [Algebra.algebraMap_eq_smul_one'] at hinj) li
                                         /-
                                           🎉 no goals
                                         -/

