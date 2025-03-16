/-- Every `x : N ⊗ M` is the image of some `y : J ⊗ M`, where `J` is a finitely generated
submodule of `N`, under the tensor product of the inclusion `J → N` and the identity `M → M`. -/
theorem exists_fg_le_eq_rTensor_subtype (x : N ⊗ M) :
    ∃ (J : Submodule R N) (_ : J.FG) (y : J ⊗ M), x = rTensor M J.subtype y := by
  induction x with
  | zero => exact ⟨⊥, fg_bot, 0, rfl⟩
  | tmul i m => exact ⟨R ∙ i, fg_span_singleton i, ⟨i, mem_span_singleton_self _⟩ ⊗ₜ[R] m, rfl⟩
  | add x₁ x₂ ihx₁ ihx₂ =>
    obtain ⟨J₁, fg₁, y₁, rfl⟩ := ihx₁
    obtain ⟨J₂, fg₂, y₂, rfl⟩ := ihx₂
    refine ⟨J₁ ⊔ J₂, fg₁.sup fg₂,
      rTensor M (J₁.inclusion le_sup_left) y₁ + rTensor M (J₂.inclusion le_sup_right) y₂, ?_⟩
    rw [map_add, ← rTensor_comp_apply, ← rTensor_comp_apply]
    rfl


theorem exists_fg_le_subset_range_rTensor_subtype (s : Set (N ⊗[R] M)) (hs : s.Finite) :
    ∃ (J : Submodule R N) (_ : J.FG), s ⊆ LinearMap.range (rTensor M J.subtype) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    s : Set (TensorProduct R N M)
    hs : s.Finite
    ⊢ Exists fun J => Exists fun x => HasSubset.Subset s ↑(LinearMap.range (Linear …
  -/
  choose J fg y eq using exists_fg_le_eq_rTensor_subtype (R := R) (M := M) (N := N)
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    s : Set (TensorProduct R N M)
    hs : s.Finite
    J : TensorProduct R N M → Submodule R N
    fg : ∀ (x : TensorProduct R N M), (J x).FG
    y : (x : TensorProduct R N M) → TensorProduct R (Subtype fun x_1 => Membership …
    eq : ∀ (x : TensorProduct R N M), Eq x ((LinearMap.rTensor M (J x).subtype) (y …
    ⊢ Exists fun J => Exists fun x => HasSubset.Subset s ↑(LinearMap.range (Linear …
  -/
  rw [← Set.finite_coe_iff] at hs
  refine ⟨⨆ x : s, J x, fg_iSup _ fun _ ↦ fg _, fun x hx ↦
    ⟨rTensor M (inclusion <| le_iSup _ ⟨x, hx⟩) (y x), .trans ?_ (eq x).symm⟩⟩
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    s : Set (TensorProduct R N M)
    hs : Finite ↑s
    J : TensorProduct R N M → Submodule R N
    fg : ∀ (x : TensorProduct R N M), (J x).FG
    y : (x : TensorProduct R N M) → TensorProduct R (Subtype fun x_1 => Membership …
    eq : ∀ (x : TensorProduct R N M), Eq x ((LinearMap.rTensor M (J x).subtype) (y …
    x : TensorProduct R N M
    hx : Membership.mem s x
    ⊢ Eq ((LinearMap.rTensor M (iSup fun x => J ↑x).subtype) ((LinearMap.rTensor M …
  -/
  rw [← comp_apply, ← rTensor_comp]; rfl
                                     /-
                                       🎉 no goals
                                     -/


/-- Every `x : I ⊗ M` is the image of some `y : J ⊗ M`, where `J ≤ I` is finitely generated,
under the tensor product of `J.inclusion ‹J ≤ I› : J → I` and the identity `M → M`. -/
theorem exists_fg_le_eq_rTensor_inclusion (x : I ⊗ M) :
    ∃ (J : Submodule R N) (_ : J.FG) (hle : J ≤ I) (y : J ⊗ M),
      x = rTensor M (J.inclusion hle) y := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    I : Submodule R N
    x : TensorProduct R (Subtype fun x => Membership.mem I x) M
    ⊢ Exists fun J => Exists fun x_1 => Exists fun hle => Exists fun y => Eq x ((L …
  -/
  obtain ⟨J, fg, y, rfl⟩ := exists_fg_le_eq_rTensor_subtype x
  /-
    case intro.intro.intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    I : Submodule R N
    J : Submodule R (Subtype fun x => Membership.mem I x)
    fg : J.FG
    y : TensorProduct R (Subtype fun x => Membership.mem J x) M
    ⊢ Exists fun J_1 => Exists fun x => Exists fun hle => Exists fun y_1 => Eq ((L …
  -/
  refine ⟨J.map I.subtype, fg.map _, I.map_subtype_le J, rTensor M (I.subtype.submoduleMap J) y, ?_⟩
  /-
    case intro.intro.intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    I : Submodule R N
    J : Submodule R (Subtype fun x => Membership.mem I x)
    fg : J.FG
    y : TensorProduct R (Subtype fun x => Membership.mem J x) M
    ⊢ Eq ((LinearMap.rTensor M J.subtype) y) ((LinearMap.rTensor M (Submodule.incl …
  -/
  rw [← LinearMap.rTensor_comp_apply]; rfl
                                       /-
                                         🎉 no goals
                                       -/


theorem exists_fg_le_subset_range_rTensor_inclusion (s : Set (I ⊗[R] M)) (hs : s.Finite) :
    ∃ (J : Submodule R N) (_ : J.FG) (hle : J ≤ I),
      s ⊆ LinearMap.range (rTensor M (J.inclusion hle)) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    I : Submodule R N
    s : Set (TensorProduct R (Subtype fun x => Membership.mem I x) M)
    hs : s.Finite
    ⊢ Exists fun J => Exists fun x => Exists fun hle => HasSubset.Subset s ↑(Linea …
  -/
  choose J fg hle y eq using exists_fg_le_eq_rTensor_inclusion (M := M) (I := I)
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    I : Submodule R N
    s : Set (TensorProduct R (Subtype fun x => Membership.mem I x) M)
    hs : s.Finite
    J : TensorProduct R (Subtype fun x => Membership.mem I x) M → Submodule R N
    fg : ∀ (x : TensorProduct R (Subtype fun x => Membership.mem I x) M), (J x).FG
    hle : ∀ (x : TensorProduct R (Subtype fun x => Membership.mem I x) M), LE.le ( …
    y : (x : TensorProduct R (Subtype fun x => Membership.mem I x) M) → TensorProd …
    eq : ∀ (x : TensorProduct R (Subtype fun x => Membership.mem I x) M), Eq x ((L …
    ⊢ Exists fun J => Exists fun x => Exists fun hle => HasSubset.Subset s ↑(Linea …
  -/
  rw [← Set.finite_coe_iff] at hs
  refine ⟨⨆ x : s, J x, fg_iSup _ fun _ ↦ fg _, iSup_le fun _ ↦ hle _, fun x hx ↦
    ⟨rTensor M (inclusion <| le_iSup _ ⟨x, hx⟩) (y x), .trans ?_ (eq x).symm⟩⟩
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    I : Submodule R N
    s : Set (TensorProduct R (Subtype fun x => Membership.mem I x) M)
    hs : Finite ↑s
    J : TensorProduct R (Subtype fun x => Membership.mem I x) M → Submodule R N
    fg : ∀ (x : TensorProduct R (Subtype fun x => Membership.mem I x) M), (J x).FG
    hle : ∀ (x : TensorProduct R (Subtype fun x => Membership.mem I x) M), LE.le ( …
    y : (x : TensorProduct R (Subtype fun x => Membership.mem I x) M) → TensorProd …
    eq : ∀ (x : TensorProduct R (Subtype fun x => Membership.mem I x) M), Eq x ((L …
    x : TensorProduct R (Subtype fun x => Membership.mem I x) M
    hx : Membership.mem s x
    ⊢ Eq ((LinearMap.rTensor M (Submodule.inclusion ⋯)) ((LinearMap.rTensor M (Sub …
  -/
  rw [← comp_apply, ← rTensor_comp]; rfl
                                     /-
                                       🎉 no goals
                                     -/


/-- Porting note: reminding Lean about this instance for Module.Finite.base_change -/
noncomputable local instance
    [CommSemiring R] [Semiring A] [Algebra R A] [AddCommMonoid M] [Module R M] :
    Module A (TensorProduct R A M) :=
  haveI : SMulCommClass R A A := IsScalarTower.to_smulCommClass
  TensorProduct.leftModule


instance Module.Finite.base_change [CommSemiring R] [Semiring A] [Algebra R A] [AddCommMonoid M]
    [Module R M] [h : Module.Finite R M] : Module.Finite A (TensorProduct R A M) := by
  classical
    obtain ⟨s, hs⟩ := h.out
    refine ⟨⟨s.image (TensorProduct.mk R A M 1), eq_top_iff.mpr ?_⟩⟩
    rintro x -
    induction x with
    | zero => exact zero_mem _
    | tmul x y =>
      -- Porting note: new TC reminder
      haveI : IsScalarTower R A (TensorProduct R A M) := TensorProduct.isScalarTower_left
      rw [Finset.coe_image, ← Submodule.span_span_of_tower R, Submodule.span_image, hs,
        Submodule.map_top, LinearMap.range_coe]
      change _ ∈ Submodule.span A (Set.range <| TensorProduct.mk R A M 1)
      rw [← mul_one x, ← smul_eq_mul, ← TensorProduct.smul_tmul']
      exact Submodule.smul_mem _ x (Submodule.subset_span <| Set.mem_range_self y)
    | add x y hx hy => exact Submodule.add_mem _ hx hy


instance Module.Finite.tensorProduct [CommSemiring R] [AddCommMonoid M] [Module R M]
    [AddCommMonoid N] [Module R N] [hM : Module.Finite R M] [hN : Module.Finite R N] :
    Module.Finite R (TensorProduct R M N) where
  out := (TensorProduct.map₂_mk_top_top_eq_top R M N).subst (hM.out.map₂ _ hN.out)


lemma Module.exists_isPrincipal_quotient_of_finite  :
    ∃ N : Submodule R M, N ≠ ⊤ ∧ Submodule.IsPrincipal (⊤ : Submodule R (M ⧸ N)) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Finite R M
    inst✝ : Nontrivial M
    ⊢ Exists fun N => And (Ne N Top.top) Top.top.IsPrincipal
  -/
  obtain ⟨n, f, hf⟩ := @Module.Finite.exists_fin R M _ _ _ _
  /-
    case intro.intro
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Finite R M
    inst✝ : Nontrivial M
    n : Nat
    f : Fin n → M
    hf : Eq (Submodule.span R (Set.range f)) Top.top
    ⊢ Exists fun N => And (Ne N Top.top) Top.top.IsPrincipal
  -/
  let s := { m : ℕ | Submodule.span R (f '' (Fin.val ⁻¹' (Set.Iio m))) ≠ ⊤ }
  have hns : ∀ x ∈ s, x < n := by
    refine fun x hx ↦ lt_iff_not_le.mpr fun e ↦ ?_
    have : (Fin.val ⁻¹' Set.Iio x : Set (Fin n)) = Set.univ := by ext y; simpa using y.2.trans_le e
    simp [s, this, hf] at hx
  /-
    case intro.intro
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Finite R M
    inst✝ : Nontrivial M
    n : Nat
    f : Fin n → M
    hf : Eq (Submodule.span R (Set.range f)) Top.top
    s : Set Nat := setOf fun m => Ne (Submodule.span R (Set.image f (Set.preimage  …
    hns : ∀ (x : Nat), Membership.mem s x → LT.lt x n
    ⊢ Exists fun N => And (Ne N Top.top) Top.top.IsPrincipal
  -/
  have hs₁ : s.Nonempty := ⟨0, by simp [s, show Set.Iio 0 = ∅ by ext; simp]⟩
  /-
    case intro.intro
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Finite R M
    inst✝ : Nontrivial M
    n : Nat
    f : Fin n → M
    hf : Eq (Submodule.span R (Set.range f)) Top.top
    s : Set Nat := setOf fun m => Ne (Submodule.span R (Set.image f (Set.preimage  …
    hns : ∀ (x : Nat), Membership.mem s x → LT.lt x n
    hs₁ : s.Nonempty
    ⊢ Exists fun N => And (Ne N Top.top) Top.top.IsPrincipal
  -/
  have hs₂ : BddAbove s := ⟨n, fun x hx ↦ (hns x hx).le⟩
  /-
    case intro.intro
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Finite R M
    inst✝ : Nontrivial M
    n : Nat
    f : Fin n → M
    hf : Eq (Submodule.span R (Set.range f)) Top.top
    s : Set Nat := setOf fun m => Ne (Submodule.span R (Set.image f (Set.preimage  …
    hns : ∀ (x : Nat), Membership.mem s x → LT.lt x n
    hs₁ : s.Nonempty
    hs₂ : BddAbove s
    ⊢ Exists fun N => And (Ne N Top.top) Top.top.IsPrincipal
  -/
  have hs := Nat.sSup_mem hs₁ hs₂
  /-
    case intro.intro
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Finite R M
    inst✝ : Nontrivial M
    n : Nat
    f : Fin n → M
    hf : Eq (Submodule.span R (Set.range f)) Top.top
    s : Set Nat := setOf fun m => Ne (Submodule.span R (Set.image f (Set.preimage  …
    hns : ∀ (x : Nat), Membership.mem s x → LT.lt x n
    hs₁ : s.Nonempty
    hs₂ : BddAbove s
    hs : Membership.mem s (SupSet.sSup s)
    ⊢ Exists fun N => And (Ne N Top.top) Top.top.IsPrincipal
  -/
  refine ⟨_, hs, ⟨⟨Submodule.mkQ _ (f ⟨_, hns _ hs⟩), ?_⟩⟩⟩
  /-
    case intro.intro
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Finite R M
    inst✝ : Nontrivial M
    n : Nat
    f : Fin n → M
    hf : Eq (Submodule.span R (Set.range f)) Top.top
    s : Set Nat := setOf fun m => Ne (Submodule.span R (Set.image f (Set.preimage  …
    hns : ∀ (x : Nat), Membership.mem s x → LT.lt x n
    hs₁ : s.Nonempty
    hs₂ : BddAbove s
    hs : Membership.mem s (SupSet.sSup s)
    ⊢ Eq Top.top (Submodule.span R (Singleton.singleton ((Submodule.span R (Set.im …
  -/
  have := not_not.mp (not_mem_of_csSup_lt (Order.lt_succ _) hs₂)
  rw [← Set.image_singleton, ← Submodule.map_span,
    ← (Submodule.comap_injective_of_surjective (Submodule.mkQ_surjective _)).eq_iff,
    Submodule.comap_map_eq, Submodule.ker_mkQ, Submodule.comap_top, ← this, ← Submodule.span_union,
    Order.Iio_succ_eq_insert (sSup s), ← Set.union_singleton, Set.preimage_union, Set.image_union,
    ← @Set.image_singleton _ _ f, Set.union_comm]
  /-
    case intro.intro
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Finite R M
    inst✝ : Nontrivial M
    n : Nat
    f : Fin n → M
    hf : Eq (Submodule.span R (Set.range f)) Top.top
    s : Set Nat := setOf fun m => Ne (Submodule.span R (Set.image f (Set.preimage  …
    hns : ∀ (x : Nat), Membership.mem s x → LT.lt x n
    hs₁ : s.Nonempty
    hs₂ : BddAbove s
    hs : Membership.mem s (SupSet.sSup s)
    this : Eq (Submodule.span R (Set.image f (Set.preimage Fin.val (Set.Iio (Order …
    ⊢ Eq (Submodule.span R (Union.union (Set.image f (Set.preimage Fin.val (Single …
  -/
  congr!
  /-
    case intro.intro.h.e'_6.h.e'_3.h.e'_4
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Finite R M
    inst✝ : Nontrivial M
    n : Nat
    f : Fin n → M
    hf : Eq (Submodule.span R (Set.range f)) Top.top
    s : Set Nat := setOf fun m => Ne (Submodule.span R (Set.image f (Set.preimage  …
    hns : ∀ (x : Nat), Membership.mem s x → LT.lt x n
    hs₁ : s.Nonempty
    hs₂ : BddAbove s
    hs : Membership.mem s (SupSet.sSup s)
    this : Eq (Submodule.span R (Set.image f (Set.preimage Fin.val (Set.Iio (Order …
    ⊢ Eq (Set.preimage Fin.val (Singleton.singleton (SupSet.sSup s))) (Singleton.s …
  -/
  ext
  /-
    case intro.intro.h.e'_6.h.e'_3.h.e'_4.h
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Finite R M
    inst✝ : Nontrivial M
    n : Nat
    f : Fin n → M
    hf : Eq (Submodule.span R (Set.range f)) Top.top
    s : Set Nat := setOf fun m => Ne (Submodule.span R (Set.image f (Set.preimage  …
    hns : ∀ (x : Nat), Membership.mem s x → LT.lt x n
    hs₁ : s.Nonempty
    hs₂ : BddAbove s
    hs : Membership.mem s (SupSet.sSup s)
    this : Eq (Submodule.span R (Set.image f (Set.preimage Fin.val (Set.Iio (Order …
    x✝ : Fin n
    ⊢ Iff (Membership.mem (Set.preimage Fin.val (Singleton.singleton (SupSet.sSup  …
  -/
  simp [Fin.ext_iff]
  /-
    🎉 no goals
  -/


lemma Module.exists_surjective_quotient_of_finite :
    ∃ (I : Ideal R) (f : M →ₗ[R] R ⧸ I), I ≠ ⊤ ∧ Function.Surjective f := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Finite R M
    inst✝ : Nontrivial M
    ⊢ Exists fun I => Exists fun f => And (Ne I Top.top) (Function.Surjective ⇑f)
  -/
  obtain ⟨N, hN, ⟨x, hx⟩⟩ := Module.exists_isPrincipal_quotient_of_finite R M
  let f := (LinearMap.toSpanSingleton R _ x).quotKerEquivOfSurjective
    (by rw [← LinearMap.range_eq_top, ← LinearMap.span_singleton_eq_range, hx])
  /-
    case intro.intro.mk.intro
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Finite R M
    inst✝ : Nontrivial M
    N : Submodule R M
    hN : Ne N Top.top
    x : HasQuotient.Quotient M N
    hx : Eq Top.top (Submodule.span R (Singleton.singleton x))
    f : LinearEquiv (RingHom.id R) (HasQuotient.Quotient R (LinearMap.ker (LinearM …
    ⊢ Exists fun I => Exists fun f => And (Ne I Top.top) (Function.Surjective ⇑f)
  -/
  refine ⟨_, f.symm.toLinearMap.comp N.mkQ, fun e ↦ ?_, f.symm.surjective.comp N.mkQ_surjective⟩
  /-
    case intro.intro.mk.intro
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Finite R M
    inst✝ : Nontrivial M
    N : Submodule R M
    hN : Ne N Top.top
    x : HasQuotient.Quotient M N
    hx : Eq Top.top (Submodule.span R (Singleton.singleton x))
    f : LinearEquiv (RingHom.id R) (HasQuotient.Quotient R (LinearMap.ker (LinearM …
    e : Eq (LinearMap.ker (LinearMap.toSpanSingleton R (HasQuotient.Quotient M N)  …
    ⊢ False
  -/
  obtain rfl : x = 0 := by simpa using LinearMap.congr_fun (LinearMap.ker_eq_top.mp e) 1
  rw [ne_eq, ← Submodule.subsingleton_quotient_iff_eq_top, ← not_nontrivial_iff_subsingleton,
    not_not] at hN
  /-
    case intro.intro.mk.intro
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Finite R M
    inst✝ : Nontrivial M
    N : Submodule R M
    hN : Nontrivial (HasQuotient.Quotient M N)
    hx : Eq Top.top (Submodule.span R (Singleton.singleton 0))
    f : LinearEquiv (RingHom.id R) (HasQuotient.Quotient R (LinearMap.ker (LinearM …
    e : Eq (LinearMap.ker (LinearMap.toSpanSingleton R (HasQuotient.Quotient M N)  …
    ⊢ False
  -/
  simp at hx
  /-
    🎉 no goals
  -/


instance : Nontrivial (M ⊗[R] M) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Finite R M
    inst✝ : Nontrivial M
    ⊢ Nontrivial (TensorProduct R M M)
  -/
  obtain ⟨I, ϕ, hI, hϕ⟩ := Module.exists_surjective_quotient_of_finite R M
  let ψ : M ⊗[R] M →ₗ[R] R ⧸ I :=
    (LinearMap.mul' R (R ⧸ I)).comp (TensorProduct.map ϕ ϕ)
  have : Nontrivial (R ⧸ I) := by
    rwa [← not_subsingleton_iff_nontrivial, Submodule.subsingleton_quotient_iff_eq_top]
  have : Function.Surjective ψ := by
    intro x; obtain ⟨x, rfl⟩ := hϕ x; obtain ⟨y, hy⟩ := hϕ 1; exact ⟨x ⊗ₜ y, by simp [ψ, hy]⟩
  /-
    case intro.intro.intro
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Finite R M
    inst✝ : Nontrivial M
    I : Ideal R
    ϕ : LinearMap (RingHom.id R) M (HasQuotient.Quotient R I)
    hI : Ne I Top.top
    hϕ : Function.Surjective ⇑ϕ
    ψ : LinearMap (RingHom.id R) (TensorProduct R M M) (HasQuotient.Quotient R I)  …
    this✝ : Nontrivial (HasQuotient.Quotient R I)
    this : Function.Surjective ⇑ψ
    ⊢ Nontrivial (TensorProduct R M M)
  -/
  exact this.nontrivial
  /-
    🎉 no goals
  -/


