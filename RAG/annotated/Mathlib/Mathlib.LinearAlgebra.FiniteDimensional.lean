/-- The dimension of a strict submodule is strictly bounded by the dimension of the ambient
space. -/
theorem finrank_lt [FiniteDimensional K V] {s : Submodule K V} (h : s < ⊤) :
    finrank K s < finrank K V := by
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    s : Submodule K V
    h : LT.lt s Top.top
    ⊢ LT.lt (Module.finrank K (Subtype fun x => Membership.mem s x)) (Module.finra …
  -/
  rw [← s.finrank_quotient_add_finrank, add_comm]
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    s : Submodule K V
    h : LT.lt s Top.top
    ⊢ LT.lt (Module.finrank K (Subtype fun x => Membership.mem s x)) (HAdd.hAdd (M …
  -/
  exact Nat.lt_add_of_pos_right (finrank_pos_iff.mpr (Quotient.nontrivial_of_lt_top _ h))
  /-
    🎉 no goals
  -/


/-- The sum of the dimensions of s + t and s ∩ t is the sum of the dimensions of s and t -/
theorem finrank_sup_add_finrank_inf_eq (s t : Submodule K V) [FiniteDimensional K s]
    [FiniteDimensional K t] :
    finrank K ↑(s ⊔ t) + finrank K ↑(s ⊓ t) = finrank K ↑s + finrank K ↑t := by
  have key : Module.rank K ↑(s ⊔ t) + Module.rank K ↑(s ⊓ t) = Module.rank K s + Module.rank K t :=
    rank_sup_add_rank_inf_eq s t
  /-
    K : Type u
    V : Type v
    inst✝⁴ : DivisionRing K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    s t : Submodule K V
    inst✝¹ : FiniteDimensional K (Subtype fun x => Membership.mem s x)
    inst✝ : FiniteDimensional K (Subtype fun x => Membership.mem t x)
    key : Eq (HAdd.hAdd (Module.rank K (Subtype fun x => Membership.mem (Max.max s …
    ⊢ Eq (HAdd.hAdd (Module.finrank K (Subtype fun x => Membership.mem (Max.max s  …
  -/
  repeat rw [← finrank_eq_rank] at key
  /-
    K : Type u
    V : Type v
    inst✝⁴ : DivisionRing K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    s t : Submodule K V
    inst✝¹ : FiniteDimensional K (Subtype fun x => Membership.mem s x)
    inst✝ : FiniteDimensional K (Subtype fun x => Membership.mem t x)
    key : Eq (HAdd.hAdd ↑(Module.finrank K (Subtype fun x => Membership.mem (Max.m …
    ⊢ Eq (HAdd.hAdd (Module.finrank K (Subtype fun x => Membership.mem (Max.max s  …
  -/
  norm_cast at key
  /-
    🎉 no goals
  -/


theorem finrank_add_le_finrank_add_finrank (s t : Submodule K V) [FiniteDimensional K s]
    [FiniteDimensional K t] : finrank K (s ⊔ t : Submodule K V) ≤ finrank K s + finrank K t := by
  /-
    K : Type u
    V : Type v
    inst✝⁴ : DivisionRing K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    s t : Submodule K V
    inst✝¹ : FiniteDimensional K (Subtype fun x => Membership.mem s x)
    inst✝ : FiniteDimensional K (Subtype fun x => Membership.mem t x)
    ⊢ LE.le (Module.finrank K (Subtype fun x => Membership.mem (Max.max s t) x)) ( …
  -/
  rw [← finrank_sup_add_finrank_inf_eq]
  /-
    K : Type u
    V : Type v
    inst✝⁴ : DivisionRing K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    s t : Submodule K V
    inst✝¹ : FiniteDimensional K (Subtype fun x => Membership.mem s x)
    inst✝ : FiniteDimensional K (Subtype fun x => Membership.mem t x)
    ⊢ LE.le (Module.finrank K (Subtype fun x => Membership.mem (Max.max s t) x)) ( …
  -/
  exact self_le_add_right _ _
  /-
    🎉 no goals
  -/


theorem finrank_add_finrank_le_of_disjoint [FiniteDimensional K V]
    {s t : Submodule K V} (hdisjoint : Disjoint s t) :
    finrank K s + finrank K t ≤ finrank K V := by
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    s t : Submodule K V
    hdisjoint : Disjoint s t
    ⊢ LE.le (HAdd.hAdd (Module.finrank K (Subtype fun x => Membership.mem s x)) (M …
  -/
  rw [← Submodule.finrank_sup_add_finrank_inf_eq s t, hdisjoint.eq_bot, finrank_bot, add_zero]
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    s t : Submodule K V
    hdisjoint : Disjoint s t
    ⊢ LE.le (Module.finrank K (Subtype fun x => Membership.mem (Max.max s t) x)) ( …
  -/
  exact Submodule.finrank_le _
  /-
    🎉 no goals
  -/


theorem eq_top_of_disjoint [FiniteDimensional K V] (s t : Submodule K V)
    (hdim : finrank K V ≤ finrank K s + finrank K t) (hdisjoint : Disjoint s t) : s ⊔ t = ⊤ := by
  have h_finrank_inf : finrank K ↑(s ⊓ t) = 0 := by
    rw [disjoint_iff_inf_le, le_bot_iff] at hdisjoint
    rw [hdisjoint, finrank_bot]
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    s t : Submodule K V
    hdim : LE.le (Module.finrank K V) (HAdd.hAdd (Module.finrank K (Subtype fun x  …
    hdisjoint : Disjoint s t
    h_finrank_inf : Eq (Module.finrank K (Subtype fun x => Membership.mem (Min.min …
    ⊢ Eq (Max.max s t) Top.top
  -/
  apply eq_top_of_finrank_eq
  replace hdim : finrank K V = finrank K s + finrank K t :=
    le_antisymm hdim (finrank_add_finrank_le_of_disjoint hdisjoint)
  /-
    case h
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    s t : Submodule K V
    hdisjoint : Disjoint s t
    h_finrank_inf : Eq (Module.finrank K (Subtype fun x => Membership.mem (Min.min …
    hdim : Eq (Module.finrank K V) (HAdd.hAdd (Module.finrank K (Subtype fun x =>  …
    ⊢ Eq (Module.finrank K (Subtype fun x => Membership.mem (Max.max s t) x)) (Mod …
  -/
  rw [hdim]
  /-
    case h
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    s t : Submodule K V
    hdisjoint : Disjoint s t
    h_finrank_inf : Eq (Module.finrank K (Subtype fun x => Membership.mem (Min.min …
    hdim : Eq (Module.finrank K V) (HAdd.hAdd (Module.finrank K (Subtype fun x =>  …
    ⊢ Eq (Module.finrank K (Subtype fun x => Membership.mem (Max.max s t) x)) (HAd …
  -/
  convert s.finrank_sup_add_finrank_inf_eq t
  /-
    case h.e'_2
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    s t : Submodule K V
    hdisjoint : Disjoint s t
    h_finrank_inf : Eq (Module.finrank K (Subtype fun x => Membership.mem (Min.min …
    hdim : Eq (Module.finrank K V) (HAdd.hAdd (Module.finrank K (Subtype fun x =>  …
    ⊢ Eq (Module.finrank K (Subtype fun x => Membership.mem (Max.max s t) x)) (HAd …
  -/
  rw [h_finrank_inf]
  /-
    case h.e'_2
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    s t : Submodule K V
    hdisjoint : Disjoint s t
    h_finrank_inf : Eq (Module.finrank K (Subtype fun x => Membership.mem (Min.min …
    hdim : Eq (Module.finrank K V) (HAdd.hAdd (Module.finrank K (Subtype fun x =>  …
    ⊢ Eq (Module.finrank K (Subtype fun x => Membership.mem (Max.max s t) x)) (HAd …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem isCompl_iff_disjoint [FiniteDimensional K V] (s t : Submodule K V)
    (hdim : finrank K V ≤ finrank K s + finrank K t) :
    IsCompl s t ↔ Disjoint s t :=
  ⟨fun h ↦ h.1, fun h ↦ ⟨h, codisjoint_iff.mpr <| eq_top_of_disjoint s t hdim h⟩⟩


/-- Given isomorphic subspaces `p q` of vector spaces `V` and `V₁` respectively,
  `p.quotient` is isomorphic to `q.quotient`. -/
noncomputable def LinearEquiv.quotEquivOfEquiv {p : Subspace K V} {q : Subspace K V₂}
    (f₁ : p ≃ₗ[K] q) (f₂ : V ≃ₗ[K] V₂) : (V ⧸ p) ≃ₗ[K] V₂ ⧸ q :=
  LinearEquiv.ofFinrankEq _ _
    (by
      rw [← @add_right_cancel_iff _ _ _ (finrank K p), Submodule.finrank_quotient_add_finrank,
        LinearEquiv.finrank_eq f₁, Submodule.finrank_quotient_add_finrank,
        LinearEquiv.finrank_eq f₂])

-- TODO: generalize to the case where one of `p` and `q` is finite-dimensional.

/-- Given the subspaces `p q`, if `p.quotient ≃ₗ[K] q`, then `q.quotient ≃ₗ[K] p` -/
noncomputable def LinearEquiv.quotEquivOfQuotEquiv {p q : Subspace K V} (f : (V ⧸ p) ≃ₗ[K] q) :
    (V ⧸ q) ≃ₗ[K] p :=
  LinearEquiv.ofFinrankEq _ _ <| by
    rw [← add_right_cancel_iff, Submodule.finrank_quotient_add_finrank, ← LinearEquiv.finrank_eq f,
      add_comm, Submodule.finrank_quotient_add_finrank]


/-- rank-nullity theorem : the dimensions of the kernel and the range of a linear map add up to
the dimension of the source space. -/
theorem finrank_range_add_finrank_ker [FiniteDimensional K V] (f : V →ₗ[K] V₂) :
    finrank K (LinearMap.range f) + finrank K (LinearMap.ker f) = finrank K V := by
  /-
    K : Type u
    V : Type v
    inst✝⁵ : DivisionRing K
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module K V
    V₂ : Type v'
    inst✝² : AddCommGroup V₂
    inst✝¹ : Module K V₂
    inst✝ : FiniteDimensional K V
    f : LinearMap (RingHom.id K) V V₂
    ⊢ Eq (HAdd.hAdd (Module.finrank K (Subtype fun x => Membership.mem (LinearMap. …
  -/
  rw [← f.quotKerEquivRange.finrank_eq]
  /-
    K : Type u
    V : Type v
    inst✝⁵ : DivisionRing K
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module K V
    V₂ : Type v'
    inst✝² : AddCommGroup V₂
    inst✝¹ : Module K V₂
    inst✝ : FiniteDimensional K V
    f : LinearMap (RingHom.id K) V V₂
    ⊢ Eq (HAdd.hAdd (Module.finrank K (HasQuotient.Quotient V (LinearMap.ker f)))  …
  -/
  exact Submodule.finrank_quotient_add_finrank _
  /-
    🎉 no goals
  -/


lemma ker_ne_bot_of_finrank_lt [FiniteDimensional K V] [FiniteDimensional K V₂] {f : V →ₗ[K] V₂}
    (h : finrank K V₂ < finrank K V) :
    LinearMap.ker f ≠ ⊥ := by
  /-
    K : Type u
    V : Type v
    inst✝⁶ : DivisionRing K
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module K V
    V₂ : Type v'
    inst✝³ : AddCommGroup V₂
    inst✝² : Module K V₂
    inst✝¹ : FiniteDimensional K V
    inst✝ : FiniteDimensional K V₂
    f : LinearMap (RingHom.id K) V V₂
    h : LT.lt (Module.finrank K V₂) (Module.finrank K V)
    ⊢ Ne (LinearMap.ker f) Bot.bot
  -/
  have h₁ := f.finrank_range_add_finrank_ker
  /-
    K : Type u
    V : Type v
    inst✝⁶ : DivisionRing K
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module K V
    V₂ : Type v'
    inst✝³ : AddCommGroup V₂
    inst✝² : Module K V₂
    inst✝¹ : FiniteDimensional K V
    inst✝ : FiniteDimensional K V₂
    f : LinearMap (RingHom.id K) V V₂
    h : LT.lt (Module.finrank K V₂) (Module.finrank K V)
    h₁ : Eq (HAdd.hAdd (Module.finrank K (Subtype fun x => Membership.mem (LinearM …
    ⊢ Ne (LinearMap.ker f) Bot.bot
  -/
  have h₂ : finrank K (LinearMap.range f) ≤ finrank K V₂ := (LinearMap.range f).finrank_le
  /-
    K : Type u
    V : Type v
    inst✝⁶ : DivisionRing K
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module K V
    V₂ : Type v'
    inst✝³ : AddCommGroup V₂
    inst✝² : Module K V₂
    inst✝¹ : FiniteDimensional K V
    inst✝ : FiniteDimensional K V₂
    f : LinearMap (RingHom.id K) V V₂
    h : LT.lt (Module.finrank K V₂) (Module.finrank K V)
    h₁ : Eq (HAdd.hAdd (Module.finrank K (Subtype fun x => Membership.mem (LinearM …
    h₂ : LE.le (Module.finrank K (Subtype fun x => Membership.mem (LinearMap.range …
    ⊢ Ne (LinearMap.ker f) Bot.bot
  -/
  suffices 0 < finrank K (LinearMap.ker f) from Submodule.one_le_finrank_iff.mp this
  /-
    K : Type u
    V : Type v
    inst✝⁶ : DivisionRing K
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module K V
    V₂ : Type v'
    inst✝³ : AddCommGroup V₂
    inst✝² : Module K V₂
    inst✝¹ : FiniteDimensional K V
    inst✝ : FiniteDimensional K V₂
    f : LinearMap (RingHom.id K) V V₂
    h : LT.lt (Module.finrank K V₂) (Module.finrank K V)
    h₁ : Eq (HAdd.hAdd (Module.finrank K (Subtype fun x => Membership.mem (LinearM …
    h₂ : LE.le (Module.finrank K (Subtype fun x => Membership.mem (LinearMap.range …
    ⊢ LT.lt 0 (Module.finrank K (Subtype fun x => Membership.mem (LinearMap.ker f) …
  -/
  omega
  /-
    🎉 no goals
  -/


theorem injective_iff_surjective_of_finrank_eq_finrank [FiniteDimensional K V]
    [FiniteDimensional K V₂] (H : finrank K V = finrank K V₂) {f : V →ₗ[K] V₂} :
    Function.Injective f ↔ Function.Surjective f := by
  /-
    K : Type u
    V : Type v
    inst✝⁶ : DivisionRing K
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module K V
    V₂ : Type v'
    inst✝³ : AddCommGroup V₂
    inst✝² : Module K V₂
    inst✝¹ : FiniteDimensional K V
    inst✝ : FiniteDimensional K V₂
    H : Eq (Module.finrank K V) (Module.finrank K V₂)
    f : LinearMap (RingHom.id K) V V₂
    ⊢ Iff (Function.Injective ⇑f) (Function.Surjective ⇑f)
  -/
  have := finrank_range_add_finrank_ker f
  /-
    K : Type u
    V : Type v
    inst✝⁶ : DivisionRing K
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module K V
    V₂ : Type v'
    inst✝³ : AddCommGroup V₂
    inst✝² : Module K V₂
    inst✝¹ : FiniteDimensional K V
    inst✝ : FiniteDimensional K V₂
    H : Eq (Module.finrank K V) (Module.finrank K V₂)
    f : LinearMap (RingHom.id K) V V₂
    this : Eq (HAdd.hAdd (Module.finrank K (Subtype fun x => Membership.mem (Linea …
    ⊢ Iff (Function.Injective ⇑f) (Function.Surjective ⇑f)
  -/
  rw [← ker_eq_bot, ← range_eq_top]; refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case refine_1
      K : Type u
      V : Type v
      inst✝⁶ : DivisionRing K
      inst✝⁵ : AddCommGroup V
      inst✝⁴ : Module K V
      V₂ : Type v'
      inst✝³ : AddCommGroup V₂
      inst✝² : Module K V₂
      inst✝¹ : FiniteDimensional K V
      inst✝ : FiniteDimensional K V₂
      H : Eq (Module.finrank K V) (Module.finrank K V₂)
      f : LinearMap (RingHom.id K) V V₂
      this : Eq (HAdd.hAdd (Module.finrank K (Subtype fun x => Membership.mem (Linea …
      h : Eq (LinearMap.ker f) Bot.bot
      ⊢ Eq (LinearMap.range f) Top.top
    -/
  · rw [h, finrank_bot, add_zero, H] at this
    /-
      case refine_1
      K : Type u
      V : Type v
      inst✝⁶ : DivisionRing K
      inst✝⁵ : AddCommGroup V
      inst✝⁴ : Module K V
      V₂ : Type v'
      inst✝³ : AddCommGroup V₂
      inst✝² : Module K V₂
      inst✝¹ : FiniteDimensional K V
      inst✝ : FiniteDimensional K V₂
      H : Eq (Module.finrank K V) (Module.finrank K V₂)
      f : LinearMap (RingHom.id K) V V₂
      this : Eq (Module.finrank K (Subtype fun x => Membership.mem (LinearMap.range  …
      h : Eq (LinearMap.ker f) Bot.bot
      ⊢ Eq (LinearMap.range f) Top.top
    -/
    exact eq_top_of_finrank_eq this
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      K : Type u
      V : Type v
      inst✝⁶ : DivisionRing K
      inst✝⁵ : AddCommGroup V
      inst✝⁴ : Module K V
      V₂ : Type v'
      inst✝³ : AddCommGroup V₂
      inst✝² : Module K V₂
      inst✝¹ : FiniteDimensional K V
      inst✝ : FiniteDimensional K V₂
      H : Eq (Module.finrank K V) (Module.finrank K V₂)
      f : LinearMap (RingHom.id K) V V₂
      this : Eq (HAdd.hAdd (Module.finrank K (Subtype fun x => Membership.mem (Linea …
      h : Eq (LinearMap.range f) Top.top
      ⊢ Eq (LinearMap.ker f) Bot.bot
    -/
  · rw [h, finrank_top, H] at this
    /-
      case refine_2
      K : Type u
      V : Type v
      inst✝⁶ : DivisionRing K
      inst✝⁵ : AddCommGroup V
      inst✝⁴ : Module K V
      V₂ : Type v'
      inst✝³ : AddCommGroup V₂
      inst✝² : Module K V₂
      inst✝¹ : FiniteDimensional K V
      inst✝ : FiniteDimensional K V₂
      H : Eq (Module.finrank K V) (Module.finrank K V₂)
      f : LinearMap (RingHom.id K) V V₂
      this : Eq (HAdd.hAdd (Module.finrank K V₂) (Module.finrank K (Subtype fun x => …
      h : Eq (LinearMap.range f) Top.top
      ⊢ Eq (LinearMap.ker f) Bot.bot
    -/
    exact Submodule.finrank_eq_zero.1 (add_right_injective _ this)
    /-
      🎉 no goals
    -/


theorem ker_eq_bot_iff_range_eq_top_of_finrank_eq_finrank [FiniteDimensional K V]
    [FiniteDimensional K V₂] (H : finrank K V = finrank K V₂) {f : V →ₗ[K] V₂} :
    LinearMap.ker f = ⊥ ↔ LinearMap.range f = ⊤ := by
  /-
    K : Type u
    V : Type v
    inst✝⁶ : DivisionRing K
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module K V
    V₂ : Type v'
    inst✝³ : AddCommGroup V₂
    inst✝² : Module K V₂
    inst✝¹ : FiniteDimensional K V
    inst✝ : FiniteDimensional K V₂
    H : Eq (Module.finrank K V) (Module.finrank K V₂)
    f : LinearMap (RingHom.id K) V V₂
    ⊢ Iff (Eq (LinearMap.ker f) Bot.bot) (Eq (LinearMap.range f) Top.top)
  -/
  rw [range_eq_top, ker_eq_bot, injective_iff_surjective_of_finrank_eq_finrank H]
  /-
    🎉 no goals
  -/


/-- Given a linear map `f` between two vector spaces with the same dimension, if
`ker f = ⊥` then `linearEquivOfInjective` is the induced isomorphism
between the two vector spaces. -/
noncomputable def linearEquivOfInjective [FiniteDimensional K V] [FiniteDimensional K V₂]
    (f : V →ₗ[K] V₂) (hf : Injective f) (hdim : finrank K V = finrank K V₂) : V ≃ₗ[K] V₂ :=
  LinearEquiv.ofBijective f
    ⟨hf, (LinearMap.injective_iff_surjective_of_finrank_eq_finrank hdim).mp hf⟩


@[simp]
theorem linearEquivOfInjective_apply [FiniteDimensional K V] [FiniteDimensional K V₂]
    {f : V →ₗ[K] V₂} (hf : Injective f) (hdim : finrank K V = finrank K V₂) (x : V) :
    f.linearEquivOfInjective hf hdim x = f x :=
  rfl


theorem finrank_lt_finrank_of_lt {s t : Submodule K V} [FiniteDimensional K t] (hst : s < t) :
    finrank K s < finrank K t :=
  (comapSubtypeEquivOfLe hst.le).finrank_eq.symm.trans_lt <|
    finrank_lt (le_top.lt_of_ne <| hst.not_le ∘ comap_subtype_eq_top.1)


theorem finrank_strictMono [FiniteDimensional K V] :
    StrictMono fun s : Submodule K V => finrank K s := fun _ _ => finrank_lt_finrank_of_lt


theorem finrank_add_eq_of_isCompl [FiniteDimensional K V] {U W : Submodule K V} (h : IsCompl U W) :
    finrank K U + finrank K W = finrank K V := by
  rw [← finrank_sup_add_finrank_inf_eq, h.codisjoint.eq_top, h.disjoint.eq_bot, finrank_bot,
    add_zero]
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    U W : Submodule K V
    h : IsCompl U W
    ⊢ Eq (Module.finrank K (Subtype fun x => Membership.mem Top.top x)) (Module.fi …
  -/
  exact finrank_top _ _
  /-
    🎉 no goals
  -/


theorem LinearIndependent.span_eq_top_of_card_eq_finrank' {ι : Type*}
    [Fintype ι] [FiniteDimensional K V] {b : ι → V} (lin_ind : LinearIndependent K b)
    (card_eq : Fintype.card ι = finrank K V) : span K (Set.range b) = ⊤ := by
  /-
    K : Type u
    V : Type v
    inst✝⁴ : DivisionRing K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : FiniteDimensional K V
    b : ι → V
    lin_ind : LinearIndependent K b
    card_eq : Eq (Fintype.card ι) (Module.finrank K V)
    ⊢ Eq (Submodule.span K (Set.range b)) Top.top
  -/
  by_contra ne_top
  /-
    K : Type u
    V : Type v
    inst✝⁴ : DivisionRing K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : FiniteDimensional K V
    b : ι → V
    lin_ind : LinearIndependent K b
    card_eq : Eq (Fintype.card ι) (Module.finrank K V)
    ne_top : Not (Eq (Submodule.span K (Set.range b)) Top.top)
    ⊢ False
  -/
  rw [← finrank_span_eq_card lin_ind] at card_eq
  /-
    K : Type u
    V : Type v
    inst✝⁴ : DivisionRing K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : FiniteDimensional K V
    b : ι → V
    lin_ind : LinearIndependent K b
    card_eq : Eq (Module.finrank K (Subtype fun x => Membership.mem (Submodule.spa …
    ne_top : Not (Eq (Submodule.span K (Set.range b)) Top.top)
    ⊢ False
  -/
  exact ne_of_lt (Submodule.finrank_lt <| lt_top_iff_ne_top.2 ne_top) card_eq
  /-
    🎉 no goals
  -/


theorem LinearIndependent.span_eq_top_of_card_eq_finrank {ι : Type*} [Nonempty ι]
    [Fintype ι] {b : ι → V} (lin_ind : LinearIndependent K b)
    (card_eq : Fintype.card ι = finrank K V) : span K (Set.range b) = ⊤ :=
  have : FiniteDimensional K V := .of_finrank_pos <| card_eq ▸ Fintype.card_pos
  lin_ind.span_eq_top_of_card_eq_finrank' card_eq


@[deprecated (since := "2024-02-14")]
alias span_eq_top_of_linearIndependent_of_card_eq_finrank :=
  LinearIndependent.span_eq_top_of_card_eq_finrank


/-- A linear independent family of `finrank K V` vectors forms a basis. -/
@[simps! repr_apply]
noncomputable def basisOfLinearIndependentOfCardEqFinrank {ι : Type*} [Nonempty ι] [Fintype ι]
    {b : ι → V} (lin_ind : LinearIndependent K b) (card_eq : Fintype.card ι = finrank K V) :
    Basis ι K V :=
  Basis.mk lin_ind <| (lin_ind.span_eq_top_of_card_eq_finrank card_eq).ge


@[simp]
theorem coe_basisOfLinearIndependentOfCardEqFinrank {ι : Type*} [Nonempty ι] [Fintype ι]
    {b : ι → V} (lin_ind : LinearIndependent K b) (card_eq : Fintype.card ι = finrank K V) :
    ⇑(basisOfLinearIndependentOfCardEqFinrank lin_ind card_eq) = b :=
  Basis.coe_mk _ _


/-- A linear independent finset of `finrank K V` vectors forms a basis. -/
@[simps! repr_apply]
noncomputable def finsetBasisOfLinearIndependentOfCardEqFinrank {s : Finset V} (hs : s.Nonempty)
    (lin_ind : LinearIndependent K ((↑) : s → V)) (card_eq : s.card = finrank K V) : Basis s K V :=
  @basisOfLinearIndependentOfCardEqFinrank _ _ _ _ _ _
    ⟨(⟨hs.choose, hs.choose_spec⟩ : s)⟩ _ _ lin_ind (_root_.trans (Fintype.card_coe _) card_eq)


@[simp]
theorem coe_finsetBasisOfLinearIndependentOfCardEqFinrank {s : Finset V} (hs : s.Nonempty)
    (lin_ind : LinearIndependent K ((↑) : s → V)) (card_eq : s.card = finrank K V) :
    ⇑(finsetBasisOfLinearIndependentOfCardEqFinrank hs lin_ind card_eq) = ((↑) : s → V) := by
  -- Porting note: added to make the next line unify the `_`s
  /-
    K : Type u
    V : Type v
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    s : Finset V
    hs : s.Nonempty
    lin_ind : LinearIndependent K Subtype.val
    card_eq : Eq s.card (Module.finrank K V)
    ⊢ Eq (⇑(finsetBasisOfLinearIndependentOfCardEqFinrank hs lin_ind card_eq)) Sub …
  -/
  rw [finsetBasisOfLinearIndependentOfCardEqFinrank]
  /-
    K : Type u
    V : Type v
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    s : Finset V
    hs : s.Nonempty
    lin_ind : LinearIndependent K Subtype.val
    card_eq : Eq s.card (Module.finrank K V)
    ⊢ Eq (⇑(basisOfLinearIndependentOfCardEqFinrank lin_ind ⋯)) Subtype.val
  -/
  exact Basis.coe_mk _ _
  /-
    🎉 no goals
  -/


/-- A linear independent set of `finrank K V` vectors forms a basis. -/
@[simps! repr_apply]
noncomputable def setBasisOfLinearIndependentOfCardEqFinrank {s : Set V} [Nonempty s] [Fintype s]
    (lin_ind : LinearIndependent K ((↑) : s → V)) (card_eq : s.toFinset.card = finrank K V) :
    Basis s K V :=
  basisOfLinearIndependentOfCardEqFinrank lin_ind (_root_.trans s.toFinset_card.symm card_eq)


@[simp]
theorem coe_setBasisOfLinearIndependentOfCardEqFinrank {s : Set V} [Nonempty s] [Fintype s]
    (lin_ind : LinearIndependent K ((↑) : s → V)) (card_eq : s.toFinset.card = finrank K V) :
    ⇑(setBasisOfLinearIndependentOfCardEqFinrank lin_ind card_eq) = ((↑) : s → V) := by
  -- Porting note: added to make the next line unify the `_`s
  /-
    K : Type u
    V : Type v
    inst✝⁴ : DivisionRing K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    s : Set V
    inst✝¹ : Nonempty ↑s
    inst✝ : Fintype ↑s
    lin_ind : LinearIndependent K Subtype.val
    card_eq : Eq s.toFinset.card (Module.finrank K V)
    ⊢ Eq (⇑(setBasisOfLinearIndependentOfCardEqFinrank lin_ind card_eq)) Subtype.val
  -/
  rw [setBasisOfLinearIndependentOfCardEqFinrank]
  /-
    K : Type u
    V : Type v
    inst✝⁴ : DivisionRing K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    s : Set V
    inst✝¹ : Nonempty ↑s
    inst✝ : Fintype ↑s
    lin_ind : LinearIndependent K Subtype.val
    card_eq : Eq s.toFinset.card (Module.finrank K V)
    ⊢ Eq (⇑(basisOfLinearIndependentOfCardEqFinrank lin_ind ⋯)) Subtype.val
  -/
  exact Basis.coe_mk _ _
  /-
    🎉 no goals
  -/


/-- Any `K`-algebra module that is 1-dimensional over `K` is simple. -/
theorem is_simple_module_of_finrank_eq_one {A} [Semiring A] [Module A V] [SMul K A]
    [IsScalarTower K A V] (h : finrank K V = 1) : IsSimpleOrder (Submodule A V) := by
  /-
    K : Type u
    V : Type v
    inst✝⁶ : DivisionRing K
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module K V
    A : Type u_1
    inst✝³ : Semiring A
    inst✝² : Module A V
    inst✝¹ : SMul K A
    inst✝ : IsScalarTower K A V
    h : Eq (Module.finrank K V) 1
    ⊢ IsSimpleOrder (Submodule A V)
  -/
  haveI := nontrivial_of_finrank_eq_succ h
  /-
    K : Type u
    V : Type v
    inst✝⁶ : DivisionRing K
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module K V
    A : Type u_1
    inst✝³ : Semiring A
    inst✝² : Module A V
    inst✝¹ : SMul K A
    inst✝ : IsScalarTower K A V
    h : Eq (Module.finrank K V) 1
    this : Nontrivial V
    ⊢ IsSimpleOrder (Submodule A V)
  -/
  refine ⟨fun S => or_iff_not_imp_left.2 fun hn => ?_⟩
  /-
    K : Type u
    V : Type v
    inst✝⁶ : DivisionRing K
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module K V
    A : Type u_1
    inst✝³ : Semiring A
    inst✝² : Module A V
    inst✝¹ : SMul K A
    inst✝ : IsScalarTower K A V
    h : Eq (Module.finrank K V) 1
    this : Nontrivial V
    S : Submodule A V
    hn : Not (Eq S Bot.bot)
    ⊢ Eq S Top.top
  -/
  rw [← restrictScalars_inj K] at hn ⊢
  /-
    K : Type u
    V : Type v
    inst✝⁶ : DivisionRing K
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module K V
    A : Type u_1
    inst✝³ : Semiring A
    inst✝² : Module A V
    inst✝¹ : SMul K A
    inst✝ : IsScalarTower K A V
    h : Eq (Module.finrank K V) 1
    this : Nontrivial V
    S : Submodule A V
    hn : Not (Eq (Submodule.restrictScalars K S) (Submodule.restrictScalars K Bot. …
    ⊢ Eq (Submodule.restrictScalars K S) (Submodule.restrictScalars K Top.top)
  -/
  haveI : FiniteDimensional _ _ := .of_finrank_eq_succ h
  /-
    K : Type u
    V : Type v
    inst✝⁶ : DivisionRing K
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module K V
    A : Type u_1
    inst✝³ : Semiring A
    inst✝² : Module A V
    inst✝¹ : SMul K A
    inst✝ : IsScalarTower K A V
    h : Eq (Module.finrank K V) 1
    this✝ : Nontrivial V
    S : Submodule A V
    hn : Not (Eq (Submodule.restrictScalars K S) (Submodule.restrictScalars K Bot. …
    this : FiniteDimensional K V
    ⊢ Eq (Submodule.restrictScalars K S) (Submodule.restrictScalars K Top.top)
  -/
  refine eq_top_of_finrank_eq ((Submodule.finrank_le _).antisymm ?_)
  /-
    K : Type u
    V : Type v
    inst✝⁶ : DivisionRing K
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module K V
    A : Type u_1
    inst✝³ : Semiring A
    inst✝² : Module A V
    inst✝¹ : SMul K A
    inst✝ : IsScalarTower K A V
    h : Eq (Module.finrank K V) 1
    this✝ : Nontrivial V
    S : Submodule A V
    hn : Not (Eq (Submodule.restrictScalars K S) (Submodule.restrictScalars K Bot. …
    this : FiniteDimensional K V
    ⊢ LE.le (Module.finrank K V) (Module.finrank K (Subtype fun x => Membership.me …
  -/
  simpa only [h, finrank_bot] using Submodule.finrank_strictMono (Ne.bot_lt hn)
  /-
    🎉 no goals
  -/


theorem Subalgebra.isSimpleOrder_of_finrank (hr : finrank F E = 2) :
    IsSimpleOrder (Subalgebra F E) :=
  let i := nontrivial_of_finrank_pos (zero_lt_two.trans_eq hr.symm)
  { toNontrivial :=
                          /-
                            F : Type u_1
                            E : Type u_2
                            inst✝² : Field F
                            inst✝¹ : Ring E
                            inst✝ : Algebra F E
                            hr : Eq (Module.finrank F E) 2
                            i : Nontrivial E := Module.nontrivial_of_finrank_pos (LT.lt.trans_eq zero_lt_t …
                            h : Eq Bot.bot Top.top
                            ⊢ False
                          -/
      ⟨⟨⊥, ⊤, fun h => by cases hr.symm.trans (Subalgebra.bot_eq_top_iff_finrank_eq_one.1 h)⟩⟩
                          /-
                            🎉 no goals
                          -/
    eq_bot_or_eq_top := by
      /-
        F : Type u_1
        E : Type u_2
        inst✝² : Field F
        inst✝¹ : Ring E
        inst✝ : Algebra F E
        hr : Eq (Module.finrank F E) 2
        i : Nontrivial E := Module.nontrivial_of_finrank_pos (LT.lt.trans_eq zero_lt_t …
        ⊢ ∀ (a : Subalgebra F E), Or (Eq a Bot.bot) (Eq a Top.top)
      -/
      intro S
      /-
        F : Type u_1
        E : Type u_2
        inst✝² : Field F
        inst✝¹ : Ring E
        inst✝ : Algebra F E
        hr : Eq (Module.finrank F E) 2
        i : Nontrivial E := Module.nontrivial_of_finrank_pos (LT.lt.trans_eq zero_lt_t …
        S : Subalgebra F E
        ⊢ Or (Eq S Bot.bot) (Eq S Top.top)
      -/
      haveI : FiniteDimensional F E := .of_finrank_eq_succ hr
      haveI : FiniteDimensional F S :=
        FiniteDimensional.finiteDimensional_submodule (Subalgebra.toSubmodule S)
      /-
        F : Type u_1
        E : Type u_2
        inst✝² : Field F
        inst✝¹ : Ring E
        inst✝ : Algebra F E
        hr : Eq (Module.finrank F E) 2
        i : Nontrivial E := Module.nontrivial_of_finrank_pos (LT.lt.trans_eq zero_lt_t …
        S : Subalgebra F E
        this✝ : FiniteDimensional F E
        this : FiniteDimensional F (Subtype fun x => Membership.mem S x)
        ⊢ Or (Eq S Bot.bot) (Eq S Top.top)
      -/
      have : finrank F S ≤ 2 := hr ▸ S.toSubmodule.finrank_le
      /-
        F : Type u_1
        E : Type u_2
        inst✝² : Field F
        inst✝¹ : Ring E
        inst✝ : Algebra F E
        hr : Eq (Module.finrank F E) 2
        i : Nontrivial E := Module.nontrivial_of_finrank_pos (LT.lt.trans_eq zero_lt_t …
        S : Subalgebra F E
        this✝¹ : FiniteDimensional F E
        this✝ : FiniteDimensional F (Subtype fun x => Membership.mem S x)
        this : LE.le (Module.finrank F (Subtype fun x => Membership.mem S x)) 2
        ⊢ Or (Eq S Bot.bot) (Eq S Top.top)
      -/
      have : 0 < finrank F S := finrank_pos_iff.mpr inferInstance
      /-
        F : Type u_1
        E : Type u_2
        inst✝² : Field F
        inst✝¹ : Ring E
        inst✝ : Algebra F E
        hr : Eq (Module.finrank F E) 2
        i : Nontrivial E := Module.nontrivial_of_finrank_pos (LT.lt.trans_eq zero_lt_t …
        S : Subalgebra F E
        this✝² : FiniteDimensional F E
        this✝¹ : FiniteDimensional F (Subtype fun x => Membership.mem S x)
        this✝ : LE.le (Module.finrank F (Subtype fun x => Membership.mem S x)) 2
        this : LT.lt 0 (Module.finrank F (Subtype fun x => Membership.mem S x))
        ⊢ Or (Eq S Bot.bot) (Eq S Top.top)
      -/
      interval_cases h : finrank F { x // x ∈ S }
        /-
          case «1»
          F : Type u_1
          E : Type u_2
          inst✝² : Field F
          inst✝¹ : Ring E
          inst✝ : Algebra F E
          hr : Eq (Module.finrank F E) 2
          i : Nontrivial E := Module.nontrivial_of_finrank_pos (LT.lt.trans_eq zero_lt_t …
          S : Subalgebra F E
          this✝² : FiniteDimensional F E
          this✝¹ : FiniteDimensional F (Subtype fun x => Membership.mem S x)
          h : Eq (Module.finrank F (Subtype fun x => Membership.mem S x)) 1
          this✝ : LE.le 1 2
          this : LT.lt 0 1
          ⊢ Or (Eq S Bot.bot) (Eq S Top.top)
        -/
      · left
        /-
          case «1».h
          F : Type u_1
          E : Type u_2
          inst✝² : Field F
          inst✝¹ : Ring E
          inst✝ : Algebra F E
          hr : Eq (Module.finrank F E) 2
          i : Nontrivial E := Module.nontrivial_of_finrank_pos (LT.lt.trans_eq zero_lt_t …
          S : Subalgebra F E
          this✝² : FiniteDimensional F E
          this✝¹ : FiniteDimensional F (Subtype fun x => Membership.mem S x)
          h : Eq (Module.finrank F (Subtype fun x => Membership.mem S x)) 1
          this✝ : LE.le 1 2
          this : LT.lt 0 1
          ⊢ Eq S Bot.bot
        -/
        exact Subalgebra.eq_bot_of_finrank_one h
        /-
          🎉 no goals
        -/
        /-
          case «2»
          F : Type u_1
          E : Type u_2
          inst✝² : Field F
          inst✝¹ : Ring E
          inst✝ : Algebra F E
          hr : Eq (Module.finrank F E) 2
          i : Nontrivial E := Module.nontrivial_of_finrank_pos (LT.lt.trans_eq zero_lt_t …
          S : Subalgebra F E
          this✝² : FiniteDimensional F E
          this✝¹ : FiniteDimensional F (Subtype fun x => Membership.mem S x)
          h : Eq (Module.finrank F (Subtype fun x => Membership.mem S x)) 2
          this✝ : LE.le 2 2
          this : LT.lt 0 2
          ⊢ Or (Eq S Bot.bot) (Eq S Top.top)
        -/
      · right
        /-
          case «2».h
          F : Type u_1
          E : Type u_2
          inst✝² : Field F
          inst✝¹ : Ring E
          inst✝ : Algebra F E
          hr : Eq (Module.finrank F E) 2
          i : Nontrivial E := Module.nontrivial_of_finrank_pos (LT.lt.trans_eq zero_lt_t …
          S : Subalgebra F E
          this✝² : FiniteDimensional F E
          this✝¹ : FiniteDimensional F (Subtype fun x => Membership.mem S x)
          h : Eq (Module.finrank F (Subtype fun x => Membership.mem S x)) 2
          this✝ : LE.le 2 2
          this : LT.lt 0 2
          ⊢ Eq S Top.top
        -/
        rw [← hr] at h
        /-
          case «2».h
          F : Type u_1
          E : Type u_2
          inst✝² : Field F
          inst✝¹ : Ring E
          inst✝ : Algebra F E
          hr : Eq (Module.finrank F E) 2
          i : Nontrivial E := Module.nontrivial_of_finrank_pos (LT.lt.trans_eq zero_lt_t …
          S : Subalgebra F E
          this✝² : FiniteDimensional F E
          this✝¹ : FiniteDimensional F (Subtype fun x => Membership.mem S x)
          h : Eq (Module.finrank F (Subtype fun x => Membership.mem S x)) (Module.finran …
          this✝ : LE.le 2 2
          this : LT.lt 0 2
          ⊢ Eq S Top.top
        -/
        rw [← Algebra.toSubmodule_eq_top]
        /-
          case «2».h
          F : Type u_1
          E : Type u_2
          inst✝² : Field F
          inst✝¹ : Ring E
          inst✝ : Algebra F E
          hr : Eq (Module.finrank F E) 2
          i : Nontrivial E := Module.nontrivial_of_finrank_pos (LT.lt.trans_eq zero_lt_t …
          S : Subalgebra F E
          this✝² : FiniteDimensional F E
          this✝¹ : FiniteDimensional F (Subtype fun x => Membership.mem S x)
          h : Eq (Module.finrank F (Subtype fun x => Membership.mem S x)) (Module.finran …
          this✝ : LE.le 2 2
          this : LT.lt 0 2
          ⊢ Eq (Subalgebra.toSubmodule S) Top.top
        -/
        exact eq_top_of_finrank_eq h }
        /-
          🎉 no goals
        -/


theorem exists_ker_pow_eq_ker_pow_succ [FiniteDimensional K V] (f : End K V) :
    ∃ k : ℕ, k ≤ finrank K V ∧ LinearMap.ker (f ^ k) = LinearMap.ker (f ^ k.succ) := by
  classical
    by_contra h_contra
    simp_rw [not_exists, not_and] at h_contra
    have h_le_ker_pow : ∀ n : ℕ, n ≤ (finrank K V).succ →
        n ≤ finrank K (LinearMap.ker (f ^ n)) := by
      intro n hn
      induction' n with n ih
      · exact zero_le (finrank _ _)
      · have h_ker_lt_ker : LinearMap.ker (f ^ n) < LinearMap.ker (f ^ n.succ) := by
          refine lt_of_le_of_ne ?_ (h_contra n (Nat.le_of_succ_le_succ hn))
          rw [pow_succ']
          apply LinearMap.ker_le_ker_comp
        have h_finrank_lt_finrank :
            finrank K (LinearMap.ker (f ^ n)) < finrank K (LinearMap.ker (f ^ n.succ)) := by
          apply Submodule.finrank_lt_finrank_of_lt h_ker_lt_ker
        calc
          n.succ ≤ (finrank K ↑(LinearMap.ker (f ^ n))).succ :=
            Nat.succ_le_succ (ih (Nat.le_of_succ_le hn))
          _ ≤ finrank K ↑(LinearMap.ker (f ^ n.succ)) := Nat.succ_le_of_lt h_finrank_lt_finrank
    have h_any_n_lt : ∀ n, n ≤ (finrank K V).succ → n ≤ finrank K V := fun n hn =>
      (h_le_ker_pow n hn).trans (Submodule.finrank_le _)
    show False
    exact Nat.not_succ_le_self _ (h_any_n_lt (finrank K V).succ (finrank K V).succ.le_refl)


theorem ker_pow_eq_ker_pow_finrank_of_le [FiniteDimensional K V] {f : End K V} {m : ℕ}
    (hm : finrank K V ≤ m) : LinearMap.ker (f ^ m) = LinearMap.ker (f ^ finrank K V) := by
  obtain ⟨k, h_k_le, hk⟩ :
    ∃ k, k ≤ finrank K V ∧ LinearMap.ker (f ^ k) = LinearMap.ker (f ^ k.succ) :=
    exists_ker_pow_eq_ker_pow_succ f
  calc
    LinearMap.ker (f ^ m) = LinearMap.ker (f ^ (k + (m - k))) := by
      rw [add_tsub_cancel_of_le (h_k_le.trans hm)]
    _ = LinearMap.ker (f ^ k) := by rw [ker_pow_constant hk _]
    _ = LinearMap.ker (f ^ (k + (finrank K V - k))) := ker_pow_constant hk (finrank K V - k)
    _ = LinearMap.ker (f ^ finrank K V) := by rw [add_tsub_cancel_of_le h_k_le]


theorem ker_pow_le_ker_pow_finrank [FiniteDimensional K V] (f : End K V) (m : ℕ) :
    LinearMap.ker (f ^ m) ≤ LinearMap.ker (f ^ finrank K V) := by
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f : Module.End K V
    m : Nat
    ⊢ LE.le (LinearMap.ker (HPow.hPow f m)) (LinearMap.ker (HPow.hPow f (Module.fi …
  -/
  by_cases h_cases : m < finrank K V
    /-
      case pos
      K : Type u
      V : Type v
      inst✝³ : DivisionRing K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : FiniteDimensional K V
      f : Module.End K V
      m : Nat
      h_cases : LT.lt m (Module.finrank K V)
      ⊢ LE.le (LinearMap.ker (HPow.hPow f m)) (LinearMap.ker (HPow.hPow f (Module.fi …
    -/
  · rw [← add_tsub_cancel_of_le (Nat.le_of_lt h_cases), add_comm, pow_add]
    /-
      case pos
      K : Type u
      V : Type v
      inst✝³ : DivisionRing K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : FiniteDimensional K V
      f : Module.End K V
      m : Nat
      h_cases : LT.lt m (Module.finrank K V)
      ⊢ LE.le (LinearMap.ker (HPow.hPow f m)) (LinearMap.ker (HMul.hMul (HPow.hPow f …
    -/
    apply LinearMap.ker_le_ker_comp
    /-
      🎉 no goals
    -/
    /-
      case neg
      K : Type u
      V : Type v
      inst✝³ : DivisionRing K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : FiniteDimensional K V
      f : Module.End K V
      m : Nat
      h_cases : Not (LT.lt m (Module.finrank K V))
      ⊢ LE.le (LinearMap.ker (HPow.hPow f m)) (LinearMap.ker (HPow.hPow f (Module.fi …
    -/
  · rw [ker_pow_eq_ker_pow_finrank_of_le (le_of_not_lt h_cases)]
    /-
      🎉 no goals
    -/


