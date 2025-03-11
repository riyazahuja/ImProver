/-- Tower law: if `A` is a `K`-module and `K` is an extension of `F` then
$\operatorname{rank}_F(A) = \operatorname{rank}_F(K) * \operatorname{rank}_K(A)$.

The universe polymorphic version of `rank_mul_rank` below. -/
theorem lift_rank_mul_lift_rank :
    Cardinal.lift.{w} (Module.rank F K) * Cardinal.lift.{v} (Module.rank K A) =
      Cardinal.lift.{v} (Module.rank F A) := by
  /-
    F : Type u
    K : Type v
    A : Type w
    inst✝¹⁰ : Ring F
    inst✝⁹ : Ring K
    inst✝⁸ : AddCommGroup A
    inst✝⁷ : Module F K
    inst✝⁶ : Module K A
    inst✝⁵ : Module F A
    inst✝⁴ : IsScalarTower F K A
    inst✝³ : StrongRankCondition F
    inst✝² : StrongRankCondition K
    inst✝¹ : Module.Free F K
    inst✝ : Module.Free K A
    ⊢ Eq (HMul.hMul (Cardinal.lift.{w, v} (Module.rank F K)) (Cardinal.lift.{v, w} …
  -/
  let b := Module.Free.chooseBasis F K
  /-
    F : Type u
    K : Type v
    A : Type w
    inst✝¹⁰ : Ring F
    inst✝⁹ : Ring K
    inst✝⁸ : AddCommGroup A
    inst✝⁷ : Module F K
    inst✝⁶ : Module K A
    inst✝⁵ : Module F A
    inst✝⁴ : IsScalarTower F K A
    inst✝³ : StrongRankCondition F
    inst✝² : StrongRankCondition K
    inst✝¹ : Module.Free F K
    inst✝ : Module.Free K A
    b : Basis (Module.Free.ChooseBasisIndex F K) F K := Module.Free.chooseBasis F K
    ⊢ Eq (HMul.hMul (Cardinal.lift.{w, v} (Module.rank F K)) (Cardinal.lift.{v, w} …
  -/
  let c := Module.Free.chooseBasis K A
  rw [← (Module.rank F K).lift_id, ← b.mk_eq_rank, ← (Module.rank K A).lift_id, ← c.mk_eq_rank,
    ← lift_umax.{w, v}, ← (b.smulTower c).mk_eq_rank, mk_prod, lift_mul, lift_lift, lift_lift,
    lift_lift, lift_lift, lift_umax.{v, w}]


/-- Tower law: if `A` is a `K`-module and `K` is an extension of `F` then
$\operatorname{rank}_F(A) = \operatorname{rank}_F(K) * \operatorname{rank}_K(A)$.

This is a simpler version of `lift_rank_mul_lift_rank` with `K` and `A` in the same universe. -/
@[stacks 09G9]
theorem rank_mul_rank (A : Type v) [AddCommGroup A]
    [Module K A] [Module F A] [IsScalarTower F K A] [Module.Free K A] :
    Module.rank F K * Module.rank K A = Module.rank F A := by
  /-
    F : Type u
    K : Type v
    inst✝¹⁰ : Ring F
    inst✝⁹ : Ring K
    inst✝⁸ : Module F K
    inst✝⁷ : StrongRankCondition F
    inst✝⁶ : StrongRankCondition K
    inst✝⁵ : Module.Free F K
    A : Type v
    inst✝⁴ : AddCommGroup A
    inst✝³ : Module K A
    inst✝² : Module F A
    inst✝¹ : IsScalarTower F K A
    inst✝ : Module.Free K A
    ⊢ Eq (HMul.hMul (Module.rank F K) (Module.rank K A)) (Module.rank F A)
  -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
  convert lift_rank_mul_lift_rank F K A <;> rw [lift_id]
                                            /-
                                              🎉 no goals
                                            -/


/-- Tower law: if `A` is a `K`-module and `K` is an extension of `F` then
$\operatorname{rank}_F(A) = \operatorname{rank}_F(K) * \operatorname{rank}_K(A)$. -/
theorem Module.finrank_mul_finrank : finrank F K * finrank K A = finrank F A := by
  /-
    F : Type u
    K : Type v
    A : Type w
    inst✝¹⁰ : Ring F
    inst✝⁹ : Ring K
    inst✝⁸ : AddCommGroup A
    inst✝⁷ : Module F K
    inst✝⁶ : Module K A
    inst✝⁵ : Module F A
    inst✝⁴ : IsScalarTower F K A
    inst✝³ : StrongRankCondition F
    inst✝² : StrongRankCondition K
    inst✝¹ : Module.Free F K
    inst✝ : Module.Free K A
    ⊢ Eq (HMul.hMul (Module.finrank F K) (Module.finrank K A)) (Module.finrank F A)
  -/
  simp_rw [finrank]
  rw [← toNat_lift.{w} (Module.rank F K), ← toNat_lift.{v} (Module.rank K A), ← toNat_mul,
    lift_rank_mul_lift_rank, toNat_lift]


/-- The rank of a free module `M` over `R` is the cardinality of `ChooseBasisIndex R M`. -/
theorem rank_eq_card_chooseBasisIndex : Module.rank R M = #(ChooseBasisIndex R M) :=
  (chooseBasis R M).mk_eq_rank''.symm


/-- The finrank of a free module `M` over `R` is the cardinality of `ChooseBasisIndex R M`. -/
theorem _root_.Module.finrank_eq_card_chooseBasisIndex [Module.Finite R M] :
    finrank R M = Fintype.card (ChooseBasisIndex R M) := by
  /-
    R : Type u
    M : Type v
    inst✝⁵ : Ring R
    inst✝⁴ : StrongRankCondition R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    ⊢ Eq (Module.finrank R M) (Fintype.card (Module.Free.ChooseBasisIndex R M))
  -/
  simp [finrank, rank_eq_card_chooseBasisIndex]
  /-
    🎉 no goals
  -/


/-- The rank of a free module `M` over an infinite scalar ring `R` is the cardinality of `M`
whenever `#R < #M`. -/
lemma rank_eq_mk_of_infinite_lt [Infinite R] (h_lt : lift.{v} #R < lift.{u} #M) :
    Module.rank R M = #M := by
  /-
    R : Type u
    M : Type v
    inst✝⁵ : Ring R
    inst✝⁴ : StrongRankCondition R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Free R M
    inst✝ : Infinite R
    h_lt : LT.lt (Cardinal.lift.{v, u} (Cardinal.mk R)) (Cardinal.lift.{u, v} (Car …
    ⊢ Eq (Module.rank R M) (Cardinal.mk M)
  -/
  have : Infinite M := infinite_iff.mpr <| lift_le.mp <| le_trans (by simp) h_lt.le
  /-
    R : Type u
    M : Type v
    inst✝⁵ : Ring R
    inst✝⁴ : StrongRankCondition R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Free R M
    inst✝ : Infinite R
    h_lt : LT.lt (Cardinal.lift.{v, u} (Cardinal.mk R)) (Cardinal.lift.{u, v} (Car …
    this : Infinite M
    ⊢ Eq (Module.rank R M) (Cardinal.mk M)
  -/
  have h : lift #M = lift #(ChooseBasisIndex R M →₀ R) := lift_mk_eq'.mpr ⟨(chooseBasis R M).repr⟩
  simp only [mk_finsupp_lift_of_infinite', lift_id', ← rank_eq_card_chooseBasisIndex, lift_max,
    lift_lift] at h
  /-
    R : Type u
    M : Type v
    inst✝⁵ : Ring R
    inst✝⁴ : StrongRankCondition R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Free R M
    inst✝ : Infinite R
    h_lt : LT.lt (Cardinal.lift.{v, u} (Cardinal.mk R)) (Cardinal.lift.{u, v} (Car …
    this : Infinite M
    h : Eq (Cardinal.lift.{max u v, v} (Cardinal.mk M)) (Max.max (Cardinal.lift.{m …
    ⊢ Eq (Module.rank R M) (Cardinal.mk M)
  -/
  refine lift_inj.mp ((max_eq_iff.mp h.symm).resolve_right <| not_and_of_not_left _ ?_).left
  /-
    R : Type u
    M : Type v
    inst✝⁵ : Ring R
    inst✝⁴ : StrongRankCondition R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Free R M
    inst✝ : Infinite R
    h_lt : LT.lt (Cardinal.lift.{v, u} (Cardinal.mk R)) (Cardinal.lift.{u, v} (Car …
    this : Infinite M
    h : Eq (Cardinal.lift.{max u v, v} (Cardinal.mk M)) (Max.max (Cardinal.lift.{m …
    ⊢ Not (Eq (Cardinal.lift.{v, u} (Cardinal.mk R)) (Cardinal.lift.{max u v, v} ( …
  -/
  exact (lift_umax.{v, u}.symm ▸ h_lt).ne
  /-
    🎉 no goals
  -/


/-- Two vector spaces are isomorphic if they have the same dimension. -/
theorem nonempty_linearEquiv_of_lift_rank_eq
    (cnd : Cardinal.lift.{v'} (Module.rank R M) = Cardinal.lift.{v} (Module.rank R M')) :
    Nonempty (M ≃ₗ[R] M') := by
  /-
    R : Type u
    M : Type v
    M' : Type v'
    inst✝⁷ : Ring R
    inst✝⁶ : StrongRankCondition R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : AddCommGroup M'
    inst✝¹ : Module R M'
    inst✝ : Module.Free R M'
    cnd : Eq (Cardinal.lift.{v', v} (Module.rank R M)) (Cardinal.lift.{v, v'} (Mod …
    ⊢ Nonempty (LinearEquiv (RingHom.id R) M M')
  -/
  obtain ⟨⟨α, B⟩⟩ := Module.Free.exists_basis (R := R) (M := M)
  /-
    case intro.mk
    R : Type u
    M : Type v
    M' : Type v'
    inst✝⁷ : Ring R
    inst✝⁶ : StrongRankCondition R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : AddCommGroup M'
    inst✝¹ : Module R M'
    inst✝ : Module.Free R M'
    cnd : Eq (Cardinal.lift.{v', v} (Module.rank R M)) (Cardinal.lift.{v, v'} (Mod …
    α : Type v
    B : Basis α R M
    ⊢ Nonempty (LinearEquiv (RingHom.id R) M M')
  -/
  obtain ⟨⟨β, B'⟩⟩ := Module.Free.exists_basis (R := R) (M := M')
  have : Cardinal.lift.{v', v} #α = Cardinal.lift.{v, v'} #β := by
    rw [B.mk_eq_rank'', cnd, B'.mk_eq_rank'']
  /-
    case intro.mk.intro.mk
    R : Type u
    M : Type v
    M' : Type v'
    inst✝⁷ : Ring R
    inst✝⁶ : StrongRankCondition R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : AddCommGroup M'
    inst✝¹ : Module R M'
    inst✝ : Module.Free R M'
    cnd : Eq (Cardinal.lift.{v', v} (Module.rank R M)) (Cardinal.lift.{v, v'} (Mod …
    α : Type v
    B : Basis α R M
    β : Type v'
    B' : Basis β R M'
    this : Eq (Cardinal.lift.{v', v} (Cardinal.mk α)) (Cardinal.lift.{v, v'} (Card …
    ⊢ Nonempty (LinearEquiv (RingHom.id R) M M')
  -/
  exact (Cardinal.lift_mk_eq.{v, v', 0}.1 this).map (B.equiv B')
  /-
    🎉 no goals
  -/


/-- Two vector spaces are isomorphic if they have the same dimension. -/
theorem nonempty_linearEquiv_of_rank_eq (cond : Module.rank R M = Module.rank R M₁) :
    Nonempty (M ≃ₗ[R] M₁) :=
  nonempty_linearEquiv_of_lift_rank_eq <| congr_arg _ cond


/-- Two vector spaces are isomorphic if they have the same dimension. -/
def LinearEquiv.ofLiftRankEq
    (cond : Cardinal.lift.{v'} (Module.rank R M) = Cardinal.lift.{v} (Module.rank R M')) :
    M ≃ₗ[R] M' :=
  Classical.choice (nonempty_linearEquiv_of_lift_rank_eq cond)


/-- Two vector spaces are isomorphic if they have the same dimension. -/
def LinearEquiv.ofRankEq (cond : Module.rank R M = Module.rank R M₁) : M ≃ₗ[R] M₁ :=
  Classical.choice (nonempty_linearEquiv_of_rank_eq cond)


/-- Two vector spaces are isomorphic if and only if they have the same dimension. -/
theorem LinearEquiv.nonempty_equiv_iff_lift_rank_eq : Nonempty (M ≃ₗ[R] M') ↔
    Cardinal.lift.{v'} (Module.rank R M) = Cardinal.lift.{v} (Module.rank R M') :=
  ⟨fun ⟨h⟩ => LinearEquiv.lift_rank_eq h, fun h => nonempty_linearEquiv_of_lift_rank_eq h⟩


/-- Two vector spaces are isomorphic if and only if they have the same dimension. -/
theorem LinearEquiv.nonempty_equiv_iff_rank_eq :
    Nonempty (M ≃ₗ[R] M₁) ↔ Module.rank R M = Module.rank R M₁ :=
  ⟨fun ⟨h⟩ => LinearEquiv.rank_eq h, fun h => nonempty_linearEquiv_of_rank_eq h⟩


/-- Two finite and free modules are isomorphic if they have the same (finite) rank. -/
theorem FiniteDimensional.nonempty_linearEquiv_of_finrank_eq
    [Module.Finite R M] [Module.Finite R M'] (cond : finrank R M = finrank R M') :
    Nonempty (M ≃ₗ[R] M') :=
                                             /-
                                               R : Type u
                                               M : Type v
                                               M' : Type v'
                                               inst✝⁹ : Ring R
                                               inst✝⁸ : StrongRankCondition R
                                               inst✝⁷ : AddCommGroup M
                                               inst✝⁶ : Module R M
                                               inst✝⁵ : Module.Free R M
                                               inst✝⁴ : AddCommGroup M'
                                               inst✝³ : Module R M'
                                               inst✝² : Module.Free R M'
                                               inst✝¹ : Module.Finite R M
                                               inst✝ : Module.Finite R M'
                                               cond : Eq (Module.finrank R M) (Module.finrank R M')
                                               ⊢ Eq (Cardinal.lift.{v', v} (Module.rank R M)) (Cardinal.lift.{v, v'} (Module. …
                                             -/
  nonempty_linearEquiv_of_lift_rank_eq <| by simp only [← finrank_eq_rank, cond, lift_natCast]
                                             /-
                                               🎉 no goals
                                             -/


/-- Two finite and free modules are isomorphic if and only if they have the same (finite) rank. -/
theorem FiniteDimensional.nonempty_linearEquiv_iff_finrank_eq [Module.Finite R M]
    [Module.Finite R M'] : Nonempty (M ≃ₗ[R] M') ↔ finrank R M = finrank R M' :=
  ⟨fun ⟨h⟩ => h.finrank_eq, fun h => nonempty_linearEquiv_of_finrank_eq h⟩


/-- Two finite and free modules are isomorphic if they have the same (finite) rank. -/
noncomputable def LinearEquiv.ofFinrankEq [Module.Finite R M] [Module.Finite R M']
    (cond : finrank R M = finrank R M') : M ≃ₗ[R] M' :=
  Classical.choice <| FiniteDimensional.nonempty_linearEquiv_of_finrank_eq cond


/-- See `rank_lt_aleph0` for the inverse direction without `Module.Free R M`. -/
lemma rank_lt_aleph0_iff : Module.rank R M < ℵ₀ ↔ Module.Finite R M := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : StrongRankCondition R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.Free R M
    ⊢ Iff (LT.lt (Module.rank R M) Cardinal.aleph0) (Module.Finite R M)
  -/
  rw [Free.rank_eq_card_chooseBasisIndex, mk_lt_aleph0_iff]
  exact ⟨fun h ↦ Finite.of_basis (Free.chooseBasis R M),
    fun I ↦ Finite.of_fintype (Free.ChooseBasisIndex R M)⟩


theorem finrank_of_not_finite (h : ¬Module.Finite R M) : finrank R M = 0 := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : StrongRankCondition R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.Free R M
    h : Not (Module.Finite R M)
    ⊢ Eq (Module.finrank R M) 0
  -/
  rw [finrank, toNat_eq_zero, ← not_lt, Module.rank_lt_aleph0_iff]
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : StrongRankCondition R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.Free R M
    h : Not (Module.Finite R M)
    ⊢ Or (Eq (Module.rank R M) 0) (Not (Module.Finite R M))
  -/
  exact .inr h
  /-
    🎉 no goals
  -/


theorem finite_of_finrank_pos (h : 0 < finrank R M) : Module.Finite R M := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : StrongRankCondition R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.Free R M
    h : LT.lt 0 (Module.finrank R M)
    ⊢ Module.Finite R M
  -/
  contrapose h
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : StrongRankCondition R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.Free R M
    h : Not (Module.Finite R M)
    ⊢ Not (LT.lt 0 (Module.finrank R M))
  -/
  simp [finrank_of_not_finite h]
  /-
    🎉 no goals
  -/


theorem finite_of_finrank_eq_succ {n : ℕ} (hn : finrank R M = n.succ) : Module.Finite R M :=
                              /-
                                R : Type u
                                M : Type v
                                inst✝⁴ : Ring R
                                inst✝³ : StrongRankCondition R
                                inst✝² : AddCommGroup M
                                inst✝¹ : Module R M
                                inst✝ : Module.Free R M
                                n : Nat
                                hn : Eq (Module.finrank R M) n.succ
                                ⊢ LT.lt 0 (Module.finrank R M)
                              -/
  finite_of_finrank_pos <| by rw [hn]; exact n.succ_pos
                                       /-
                                         🎉 no goals
                                       -/


theorem finite_iff_of_rank_eq_nsmul {W} [AddCommGroup W] [Module R W] [Module.Free R W] {n : ℕ}
    (hn : n ≠ 0) (hVW : Module.rank R M = n • Module.rank R W) :
    Module.Finite R M ↔ Module.Finite R W := by
  /-
    R : Type u
    M : Type v
    inst✝⁷ : Ring R
    inst✝⁶ : StrongRankCondition R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    W : Type v
    inst✝² : AddCommGroup W
    inst✝¹ : Module R W
    inst✝ : Module.Free R W
    n : Nat
    hn : Ne n 0
    hVW : Eq (Module.rank R M) (HSMul.hSMul n (Module.rank R W))
    ⊢ Iff (Module.Finite R M) (Module.Finite R W)
  -/
  simp only [← rank_lt_aleph0_iff, hVW, nsmul_lt_aleph0_iff_of_ne_zero hn]
  /-
    🎉 no goals
  -/


/-- A finite rank free module has a basis indexed by `Fin (finrank R M)`. -/
noncomputable def finBasis [Module.Finite R M] :
    Basis (Fin (finrank R M)) R M :=
  (Module.Free.chooseBasis R M).reindex (Fintype.equivFinOfCardEq
    (finrank_eq_card_chooseBasisIndex R M).symm)


/-- A rank `n` free module has a basis indexed by `Fin n`. -/
noncomputable def finBasisOfFinrankEq [Module.Finite R M] {n : ℕ} (hn : finrank R M = n) :
    Basis (Fin n) R M := (finBasis R M).reindex (finCongr hn)


/-- A free module with rank 1 has a basis with one element. -/
noncomputable def basisUnique (ι : Type*) [Unique ι]
    (h : finrank R M = 1) :
    Basis ι R M :=
  haveI : Module.Finite R M :=
    Module.finite_of_finrank_pos (_root_.zero_lt_one.trans_le h.symm.le)
  (finBasisOfFinrankEq R M h).reindex (Equiv.ofUnique _ _)


@[simp]
theorem basisUnique_repr_eq_zero_iff {ι : Type*} [Unique ι]
    {h : finrank R M = 1} {v : M} {i : ι} :
    (basisUnique ι h).repr v i = 0 ↔ v = 0 :=
  ⟨fun hv =>
    (basisUnique ι h).repr.map_eq_zero_iff.mp (Finsupp.ext fun j => Subsingleton.elim i j ▸ hv),
                 /-
                   R : Type u
                   M : Type v
                   inst✝⁵ : Ring R
                   inst✝⁴ : StrongRankCondition R
                   inst✝³ : AddCommGroup M
                   inst✝² : Module R M
                   inst✝¹ : Module.Free R M
                   ι : Type u_1
                   inst✝ : Unique ι
                   h : Eq (Module.finrank R M) 1
                   v : M
                   i : ι
                   hv : Eq v 0
                   ⊢ Eq (((Module.basisUnique ι h).repr v) i) 0
                 -/
    fun hv => by rw [hv, LinearEquiv.map_zero, Finsupp.zero_apply]⟩
                 /-
                   🎉 no goals
                 -/


