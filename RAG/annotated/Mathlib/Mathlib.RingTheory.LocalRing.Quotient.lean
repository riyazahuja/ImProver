local notation "p" => maximalIdeal R

local notation "pS" => Ideal.map (algebraMap R S) p


theorem quotient_span_eq_top_iff_span_eq_top (s : Set S) :
    span (R ⧸ p) ((Ideal.Quotient.mk (I := pS)) '' s) = ⊤ ↔ span R s = ⊤ := by
  have H : (span (R ⧸ p) ((Ideal.Quotient.mk (I := pS)) '' s)).restrictScalars R =
      (span R s).map (IsScalarTower.toAlgHom R S (S ⧸ pS)) := by
    rw [map_span, ← restrictScalars_span R (R ⧸ p) Ideal.Quotient.mk_surjective,
      IsScalarTower.coe_toAlgHom', Ideal.Quotient.algebraMap_eq]
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : IsLocalRing R
    inst✝ : Module.Finite R S
    s : Set S
    H : Eq (Submodule.restrictScalars R (Submodule.span (HasQuotient.Quotient R (I …
    ⊢ Iff (Eq (Submodule.span (HasQuotient.Quotient R (IsLocalRing.maximalIdeal R) …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : IsLocalRing R
      inst✝ : Module.Finite R S
      s : Set S
      H : Eq (Submodule.restrictScalars R (Submodule.span (HasQuotient.Quotient R (I …
      ⊢ Eq (Submodule.span (HasQuotient.Quotient R (IsLocalRing.maximalIdeal R)) (Se …
    -/
  · intro hs
    /-
      case mp
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : IsLocalRing R
      inst✝ : Module.Finite R S
      s : Set S
      H : Eq (Submodule.restrictScalars R (Submodule.span (HasQuotient.Quotient R (I …
      hs : Eq (Submodule.span (HasQuotient.Quotient R (IsLocalRing.maximalIdeal R))  …
      ⊢ Eq (Submodule.span R s) Top.top
    -/
    rw [← top_le_iff]
    /-
      case mp
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : IsLocalRing R
      inst✝ : Module.Finite R S
      s : Set S
      H : Eq (Submodule.restrictScalars R (Submodule.span (HasQuotient.Quotient R (I …
      hs : Eq (Submodule.span (HasQuotient.Quotient R (IsLocalRing.maximalIdeal R))  …
      ⊢ LE.le Top.top (Submodule.span R s)
    -/
    apply le_of_le_smul_of_le_jacobson_bot
      /-
        case mp.hN'
        R : Type u_1
        S : Type u_2
        inst✝⁴ : CommRing R
        inst✝³ : CommRing S
        inst✝² : Algebra R S
        inst✝¹ : IsLocalRing R
        inst✝ : Module.Finite R S
        s : Set S
        H : Eq (Submodule.restrictScalars R (Submodule.span (HasQuotient.Quotient R (I …
        hs : Eq (Submodule.span (HasQuotient.Quotient R (IsLocalRing.maximalIdeal R))  …
        ⊢ Top.top.FG
      -/
    · exact Module.finite_def.mp ‹_›
      /-
        🎉 no goals
      -/
      /-
        case mp.hIJ
        R : Type u_1
        S : Type u_2
        inst✝⁴ : CommRing R
        inst✝³ : CommRing S
        inst✝² : Algebra R S
        inst✝¹ : IsLocalRing R
        inst✝ : Module.Finite R S
        s : Set S
        H : Eq (Submodule.restrictScalars R (Submodule.span (HasQuotient.Quotient R (I …
        hs : Eq (Submodule.span (HasQuotient.Quotient R (IsLocalRing.maximalIdeal R))  …
        ⊢ LE.le ?mp.I Bot.bot.jacobson
      -/
    · exact (jacobson_eq_maximalIdeal ⊥ bot_ne_top).ge
      /-
        🎉 no goals
      -/
      /-
        case mp.hNN
        R : Type u_1
        S : Type u_2
        inst✝⁴ : CommRing R
        inst✝³ : CommRing S
        inst✝² : Algebra R S
        inst✝¹ : IsLocalRing R
        inst✝ : Module.Finite R S
        s : Set S
        H : Eq (Submodule.restrictScalars R (Submodule.span (HasQuotient.Quotient R (I …
        hs : Eq (Submodule.span (HasQuotient.Quotient R (IsLocalRing.maximalIdeal R))  …
        ⊢ LE.le Top.top (Max.max (Submodule.span R s) (HSMul.hSMul (IsLocalRing.maxima …
      -/
    · rw [Ideal.smul_top_eq_map]
      /-
        case mp.hNN
        R : Type u_1
        S : Type u_2
        inst✝⁴ : CommRing R
        inst✝³ : CommRing S
        inst✝² : Algebra R S
        inst✝¹ : IsLocalRing R
        inst✝ : Module.Finite R S
        s : Set S
        H : Eq (Submodule.restrictScalars R (Submodule.span (HasQuotient.Quotient R (I …
        hs : Eq (Submodule.span (HasQuotient.Quotient R (IsLocalRing.maximalIdeal R))  …
        ⊢ LE.le Top.top (Max.max (Submodule.span R s) (Submodule.restrictScalars R (Id …
      -/
      rintro x -
      have : LinearMap.ker (IsScalarTower.toAlgHom R S (S ⧸ pS)) =
          restrictScalars R pS := by
        ext; simp [Ideal.Quotient.eq_zero_iff_mem]
      /-
        case mp.hNN
        R : Type u_1
        S : Type u_2
        inst✝⁴ : CommRing R
        inst✝³ : CommRing S
        inst✝² : Algebra R S
        inst✝¹ : IsLocalRing R
        inst✝ : Module.Finite R S
        s : Set S
        H : Eq (Submodule.restrictScalars R (Submodule.span (HasQuotient.Quotient R (I …
        hs : Eq (Submodule.span (HasQuotient.Quotient R (IsLocalRing.maximalIdeal R))  …
        x : S
        this : Eq (LinearMap.ker (IsScalarTower.toAlgHom R S (HasQuotient.Quotient S ( …
        ⊢ Membership.mem (Max.max (Submodule.span R s) (Submodule.restrictScalars R (I …
      -/
      rw [← this, ← comap_map_eq, mem_comap, ← H, hs, restrictScalars_top]
      /-
        case mp.hNN
        R : Type u_1
        S : Type u_2
        inst✝⁴ : CommRing R
        inst✝³ : CommRing S
        inst✝² : Algebra R S
        inst✝¹ : IsLocalRing R
        inst✝ : Module.Finite R S
        s : Set S
        H : Eq (Submodule.restrictScalars R (Submodule.span (HasQuotient.Quotient R (I …
        hs : Eq (Submodule.span (HasQuotient.Quotient R (IsLocalRing.maximalIdeal R))  …
        x : S
        this : Eq (LinearMap.ker (IsScalarTower.toAlgHom R S (HasQuotient.Quotient S ( …
        ⊢ Membership.mem Top.top ((IsScalarTower.toAlgHom R S (HasQuotient.Quotient S  …
      -/
      exact mem_top
      /-
        🎉 no goals
      -/
    /-
      case mpr
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : IsLocalRing R
      inst✝ : Module.Finite R S
      s : Set S
      H : Eq (Submodule.restrictScalars R (Submodule.span (HasQuotient.Quotient R (I …
      ⊢ Eq (Submodule.span R s) Top.top → Eq (Submodule.span (HasQuotient.Quotient R …
    -/
  · intro hs
    rwa [hs, Submodule.map_top, LinearMap.range_eq_top.mpr,
      restrictScalars_eq_top_iff] at H
    /-
      case mpr
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : IsLocalRing R
      inst✝ : Module.Finite R S
      s : Set S
      H : Eq (Submodule.restrictScalars R (Submodule.span (HasQuotient.Quotient R (I …
      hs : Eq (Submodule.span R s) Top.top
      ⊢ Function.Surjective ⇑(IsScalarTower.toAlgHom R S (HasQuotient.Quotient S (Id …
    -/
    rw [IsScalarTower.coe_toAlgHom', Ideal.Quotient.algebraMap_eq]
    /-
      case mpr
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : IsLocalRing R
      inst✝ : Module.Finite R S
      s : Set S
      H : Eq (Submodule.restrictScalars R (Submodule.span (HasQuotient.Quotient R (I …
      hs : Eq (Submodule.span R s) Top.top
      ⊢ Function.Surjective ⇑(Ideal.Quotient.mk (Ideal.map (algebraMap R S) (IsLocal …
    -/
    exact Ideal.Quotient.mk_surjective
    /-
      🎉 no goals
    -/


theorem finrank_quotient_map :
    finrank (R ⧸ p) (S ⧸ pS) = finrank R S := by
  classical
  have : Module.Finite R (S ⧸ pS) := Module.Finite.of_surjective
    (IsScalarTower.toAlgHom R S (S ⧸ pS)).toLinearMap (Ideal.Quotient.mk_surjective (I := pS))
  have : Module.Finite (R ⧸ p) (S ⧸ pS) := Module.Finite.of_restrictScalars_finite R _ _
  apply le_antisymm
  · let b := Module.Free.chooseBasis R S
    conv_rhs => rw [finrank_eq_card_chooseBasisIndex]
    apply finrank_le_of_span_eq_top
    rw [Set.range_comp]
    apply (quotient_span_eq_top_iff_span_eq_top _).mpr b.span_eq
  · let b := Module.Free.chooseBasis (R ⧸ p) (S ⧸ pS)
    choose b' hb' using fun i ↦ Ideal.Quotient.mk_surjective (b i)
    conv_rhs => rw [finrank_eq_card_chooseBasisIndex]
    refine finrank_le_of_span_eq_top (v := b') ?_
    apply (quotient_span_eq_top_iff_span_eq_top _).mp
    rw [← Set.range_comp, show Ideal.Quotient.mk pS ∘ b' = ⇑b from funext hb']
    exact b.span_eq


/-- Given a basis of `S`, the induced basis of `S / Ideal.map (algebraMap R S) p`. -/
noncomputable
def basisQuotient [Fintype ι] (b : Basis ι R S) : Basis ι (R ⧸ p) (S ⧸ pS) :=
  basisOfTopLeSpanOfCardEqFinrank (Ideal.Quotient.mk pS ∘ b)
    (by
      /-
        R : Type u_1
        S : Type u_2
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        inst✝³ : IsLocalRing R
        inst✝² : Module.Finite R S
        inst✝¹ : Module.Free R S
        ι : Type u_3
        inst✝ : Fintype ι
        b : Basis ι R S
        ⊢ LE.le Top.top (Submodule.span (HasQuotient.Quotient R (IsLocalRing.maximalId …
      -/
      rw [Set.range_comp]
      /-
        R : Type u_1
        S : Type u_2
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        inst✝³ : IsLocalRing R
        inst✝² : Module.Finite R S
        inst✝¹ : Module.Free R S
        ι : Type u_3
        inst✝ : Fintype ι
        b : Basis ι R S
        ⊢ LE.le Top.top (Submodule.span (HasQuotient.Quotient R (IsLocalRing.maximalId …
      -/
      exact ((quotient_span_eq_top_iff_span_eq_top _).mpr b.span_eq).ge)
      /-
        🎉 no goals
      -/
        /-
          R : Type u_1
          S : Type u_2
          inst✝⁶ : CommRing R
          inst✝⁵ : CommRing S
          inst✝⁴ : Algebra R S
          inst✝³ : IsLocalRing R
          inst✝² : Module.Finite R S
          inst✝¹ : Module.Free R S
          ι : Type u_3
          inst✝ : Fintype ι
          b : Basis ι R S
          ⊢ Eq (Fintype.card ι) (Module.finrank (HasQuotient.Quotient R (IsLocalRing.max …
        -/
    (by rw [finrank_quotient_map, finrank_eq_card_basis b])
        /-
          🎉 no goals
        -/


lemma basisQuotient_apply [Fintype ι] (b : Basis ι R S) (i) :
    (basisQuotient b) i = Ideal.Quotient.mk pS (b i) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    inst✝³ : IsLocalRing R
    inst✝² : Module.Finite R S
    inst✝¹ : Module.Free R S
    ι : Type u_3
    inst✝ : Fintype ι
    b : Basis ι R S
    i : ι
    ⊢ Eq ((IsLocalRing.basisQuotient b) i) ((Ideal.Quotient.mk (Ideal.map (algebra …
  -/
  delta basisQuotient
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    inst✝³ : IsLocalRing R
    inst✝² : Module.Finite R S
    inst✝¹ : Module.Free R S
    ι : Type u_3
    inst✝ : Fintype ι
    b : Basis ι R S
    i : ι
    ⊢ Eq ((basisOfTopLeSpanOfCardEqFinrank (Function.comp ⇑(Ideal.Quotient.mk (Ide …
  -/
  rw [coe_basisOfTopLeSpanOfCardEqFinrank, Function.comp_apply]
  /-
    🎉 no goals
  -/


lemma basisQuotient_repr {ι} [Fintype ι] (b : Basis ι R S) (x) (i) :
    (basisQuotient b).repr (Ideal.Quotient.mk pS x) i =
    Ideal.Quotient.mk p (b.repr x i) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    inst✝³ : IsLocalRing R
    inst✝² : Module.Finite R S
    inst✝¹ : Module.Free R S
    ι : Type u_4
    inst✝ : Fintype ι
    b : Basis ι R S
    x : S
    i : ι
    ⊢ Eq (((IsLocalRing.basisQuotient b).repr ((Ideal.Quotient.mk (Ideal.map (alge …
  -/
  refine congr_fun (g := Ideal.Quotient.mk p ∘ b.repr x) ?_ i
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    inst✝³ : IsLocalRing R
    inst✝² : Module.Finite R S
    inst✝¹ : Module.Free R S
    ι : Type u_4
    inst✝ : Fintype ι
    b : Basis ι R S
    x : S
    i : ι
    ⊢ Eq (⇑((IsLocalRing.basisQuotient b).repr ((Ideal.Quotient.mk (Ideal.map (alg …
  -/
  apply (Finsupp.linearEquivFunOnFinite (R ⧸ p) _ _).symm.injective
  /-
    case a
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    inst✝³ : IsLocalRing R
    inst✝² : Module.Finite R S
    inst✝¹ : Module.Free R S
    ι : Type u_4
    inst✝ : Fintype ι
    b : Basis ι R S
    x : S
    i : ι
    ⊢ Eq ((Finsupp.linearEquivFunOnFinite (HasQuotient.Quotient R (IsLocalRing.max …
  -/
  apply (basisQuotient b).repr.symm.injective
  simp only [Finsupp.linearEquivFunOnFinite_symm_coe, LinearEquiv.symm_apply_apply,
    Basis.repr_symm_apply]
  rw [Finsupp.linearCombination_eq_fintype_linearCombination_apply _ (R ⧸ p),
    Fintype.linearCombination_apply]
  simp only [Function.comp_apply, basisQuotient_apply,
    Ideal.Quotient.mk_smul_mk_quotient_map_quotient, ← Algebra.smul_def]
  /-
    case a.a
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    inst✝³ : IsLocalRing R
    inst✝² : Module.Finite R S
    inst✝¹ : Module.Free R S
    ι : Type u_4
    inst✝ : Fintype ι
    b : Basis ι R S
    x : S
    i : ι
    ⊢ Eq ((Ideal.Quotient.mk (Ideal.map (algebraMap R S) (IsLocalRing.maximalIdeal …
  -/
  rw [← map_sum, Basis.sum_repr b x]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-11")]
alias LocalRing.quotient_span_eq_top_iff_span_eq_top :=
  IsLocalRing.quotient_span_eq_top_iff_span_eq_top


@[deprecated (since := "2024-11-11")]
alias LocalRing.finrank_quotient_map := IsLocalRing.finrank_quotient_map


@[deprecated (since := "2024-11-11")]
alias LocalRing.basisQuotient := IsLocalRing.basisQuotient


@[deprecated (since := "2024-11-11")]
alias LocalRing.basisQuotient_apply := IsLocalRing.basisQuotient_apply


@[deprecated (since := "2024-11-11")]
alias LocalRing.basisQuotient_repr := IsLocalRing.basisQuotient_repr

