open UniqueFactorizationMonoid in
/-- Every non-zero prime ideal in a unique factorization domain contains a prime element. -/
theorem Ideal.IsPrime.exists_mem_prime_of_ne_bot {R : Type*} [CommSemiring R] [IsDomain R]
    [UniqueFactorizationMonoid R] {I : Ideal R} (hI₂ : I.IsPrime) (hI : I ≠ ⊥) :
    ∃ x ∈ I, Prime x := by
  classical
  obtain ⟨a : R, ha₁ : a ∈ I, ha₂ : a ≠ 0⟩ := Submodule.exists_mem_ne_zero_of_ne_bot hI
  replace ha₁ : (factors a).prod ∈ I := by
    obtain ⟨u : Rˣ, hu : (factors a).prod * u = a⟩ := factors_prod ha₂
    rwa [← hu, mul_unit_mem_iff_mem _ u.isUnit] at ha₁
  obtain ⟨p : R, hp₁ : p ∈ factors a, hp₂ : p ∈ I⟩ :=
    (hI₂.multiset_prod_mem_iff_exists_mem <| factors a).1 ha₁
  exact ⟨p, hp₂, prime_of_factor p hp₁⟩


/-- The ascending chain condition on principal ideals holds in a `WfDvdMonoid` domain. -/
lemma Ideal.setOf_isPrincipal_wellFoundedOn_gt [CommSemiring α] [WfDvdMonoid α] [IsDomain α] :
    {I : Ideal α | I.IsPrincipal}.WellFoundedOn (· > ·) := by
  have : {I : Ideal α | I.IsPrincipal} = ((fun a ↦ Ideal.span {a}) '' Set.univ) := by
    ext
    simp [Submodule.isPrincipal_iff, eq_comm]
  /-
    α : Type u_1
    inst✝² : CommSemiring α
    inst✝¹ : WfDvdMonoid α
    inst✝ : IsDomain α
    this : Eq (setOf fun I => Submodule.IsPrincipal I) (Set.image (fun a => Ideal. …
    ⊢ (setOf fun I => Submodule.IsPrincipal I).WellFoundedOn fun x1 x2 => GT.gt x1 …
  -/
  rw [this, Set.wellFoundedOn_image, Set.wellFoundedOn_univ]
  /-
    α : Type u_1
    inst✝² : CommSemiring α
    inst✝¹ : WfDvdMonoid α
    inst✝ : IsDomain α
    this : Eq (setOf fun I => Submodule.IsPrincipal I) (Set.image (fun a => Ideal. …
    ⊢ WellFounded (Function.onFun (fun x1 x2 => GT.gt x1 x2) fun a => Ideal.span ( …
  -/
  convert wellFounded_dvdNotUnit (α := α)
  /-
    case h.e'_2
    α : Type u_1
    inst✝² : CommSemiring α
    inst✝¹ : WfDvdMonoid α
    inst✝ : IsDomain α
    this : Eq (setOf fun I => Submodule.IsPrincipal I) (Set.image (fun a => Ideal. …
    ⊢ Eq (Function.onFun (fun x1 x2 => GT.gt x1 x2) fun a => Ideal.span (Singleton …
  -/
  ext
  /-
    case h.e'_2.h.h.a
    α : Type u_1
    inst✝² : CommSemiring α
    inst✝¹ : WfDvdMonoid α
    inst✝ : IsDomain α
    this : Eq (setOf fun I => Submodule.IsPrincipal I) (Set.image (fun a => Ideal. …
    x✝¹ x✝ : α
    ⊢ Iff (Function.onFun (fun x1 x2 => GT.gt x1 x2) (fun a => Ideal.span (Singlet …
  -/
  exact Ideal.span_singleton_lt_span_singleton
  /-
    🎉 no goals
  -/


/-- The ascending chain condition on principal ideals in a domain is sufficient to prove that
the domain is `WfDvdMonoid`. -/
lemma WfDvdMonoid.of_setOf_isPrincipal_wellFoundedOn_gt [CommSemiring α] [IsDomain α]
    (h : {I : Ideal α | I.IsPrincipal}.WellFoundedOn (· > ·)) :
    WfDvdMonoid α := by
  /-
    α : Type u_1
    inst✝¹ : CommSemiring α
    inst✝ : IsDomain α
    h : (setOf fun I => Submodule.IsPrincipal I).WellFoundedOn fun x1 x2 => GT.gt  …
    ⊢ WfDvdMonoid α
  -/
  have : WellFounded (α := {I : Ideal α // I.IsPrincipal}) (· > ·) := h
  /-
    α : Type u_1
    inst✝¹ : CommSemiring α
    inst✝ : IsDomain α
    h : (setOf fun I => Submodule.IsPrincipal I).WellFoundedOn fun x1 x2 => GT.gt  …
    this : WellFounded fun x1 x2 => GT.gt x1 x2
    ⊢ WfDvdMonoid α
  -/
  constructor
  /-
    case wf
    α : Type u_1
    inst✝¹ : CommSemiring α
    inst✝ : IsDomain α
    h : (setOf fun I => Submodule.IsPrincipal I).WellFoundedOn fun x1 x2 => GT.gt  …
    this : WellFounded fun x1 x2 => GT.gt x1 x2
    ⊢ WellFounded DvdNotUnit
  -/
  convert InvImage.wf (fun a => ⟨Ideal.span ({a} : Set α), _, rfl⟩) this
  /-
    case h.e'_2
    α : Type u_1
    inst✝¹ : CommSemiring α
    inst✝ : IsDomain α
    h : (setOf fun I => Submodule.IsPrincipal I).WellFoundedOn fun x1 x2 => GT.gt  …
    this : WellFounded fun x1 x2 => GT.gt x1 x2
    ⊢ Eq DvdNotUnit (InvImage (fun x1 x2 => GT.gt x1 x2) fun a => ⟨Ideal.span (Sin …
  -/
  ext
  /-
    case h.e'_2.h.h.a
    α : Type u_1
    inst✝¹ : CommSemiring α
    inst✝ : IsDomain α
    h : (setOf fun I => Submodule.IsPrincipal I).WellFoundedOn fun x1 x2 => GT.gt  …
    this : WellFounded fun x1 x2 => GT.gt x1 x2
    x✝¹ x✝ : α
    ⊢ Iff (DvdNotUnit x✝¹ x✝) (InvImage (fun x1 x2 => GT.gt x1 x2) (fun a => ⟨Idea …
  -/
  exact Ideal.span_singleton_lt_span_singleton.symm
  /-
    🎉 no goals
  -/


