/-- **Nakayama's Lemma** - A slightly more general version of (2) in
[Stacks 00DV](https://stacks.math.columbia.edu/tag/00DV).
See also `eq_bot_of_le_smul_of_le_jacobson_bot` for the special case when `J = ⊥`. -/
@[stacks 00DV "(2)"]
theorem eq_smul_of_le_smul_of_le_jacobson {I J : Ideal R} {N : Submodule R M} (hN : N.FG)
    (hIN : N ≤ I • N) (hIjac : I ≤ jacobson J) : N = J • N := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I J : Ideal R
    N : Submodule R M
    hN : N.FG
    hIN : LE.le N (HSMul.hSMul I N)
    hIjac : LE.le I J.jacobson
    ⊢ Eq N (HSMul.hSMul J N)
  -/
  refine le_antisymm ?_ (Submodule.smul_le.2 fun _ _ _ => Submodule.smul_mem _ _)
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I J : Ideal R
    N : Submodule R M
    hN : N.FG
    hIN : LE.le N (HSMul.hSMul I N)
    hIjac : LE.le I J.jacobson
    ⊢ LE.le N (HSMul.hSMul J N)
  -/
  intro n hn
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I J : Ideal R
    N : Submodule R M
    hN : N.FG
    hIN : LE.le N (HSMul.hSMul I N)
    hIjac : LE.le I J.jacobson
    n : M
    hn : Membership.mem N n
    ⊢ Membership.mem (HSMul.hSMul J N) n
  -/
  cases' Submodule.exists_sub_one_mem_and_smul_eq_zero_of_fg_of_le_smul I N hN hIN with r hr
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I J : Ideal R
    N : Submodule R M
    hN : N.FG
    hIN : LE.le N (HSMul.hSMul I N)
    hIjac : LE.le I J.jacobson
    n : M
    hn : Membership.mem N n
    r : R
    hr : And (Membership.mem I (HSub.hSub r 1)) (∀ (n : M), Membership.mem N n → E …
    ⊢ Membership.mem (HSMul.hSMul J N) n
  -/
  cases' exists_mul_sub_mem_of_sub_one_mem_jacobson r (hIjac hr.1) with s hs
  have : n = -(s * r - 1) • n := by
    rw [neg_sub, sub_smul, mul_smul, hr.2 n hn, one_smul, smul_zero, sub_zero]
  /-
    case intro.intro
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I J : Ideal R
    N : Submodule R M
    hN : N.FG
    hIN : LE.le N (HSMul.hSMul I N)
    hIjac : LE.le I J.jacobson
    n : M
    hn : Membership.mem N n
    r : R
    hr : And (Membership.mem I (HSub.hSub r 1)) (∀ (n : M), Membership.mem N n → E …
    s : R
    hs : Membership.mem J (HSub.hSub (HMul.hMul s r) 1)
    this : Eq n (HSMul.hSMul (Neg.neg (HSub.hSub (HMul.hMul s r) 1)) n)
    ⊢ Membership.mem (HSMul.hSMul J N) n
  -/
  rw [this]
  /-
    case intro.intro
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I J : Ideal R
    N : Submodule R M
    hN : N.FG
    hIN : LE.le N (HSMul.hSMul I N)
    hIjac : LE.le I J.jacobson
    n : M
    hn : Membership.mem N n
    r : R
    hr : And (Membership.mem I (HSub.hSub r 1)) (∀ (n : M), Membership.mem N n → E …
    s : R
    hs : Membership.mem J (HSub.hSub (HMul.hMul s r) 1)
    this : Eq n (HSMul.hSMul (Neg.neg (HSub.hSub (HMul.hMul s r) 1)) n)
    ⊢ Membership.mem (HSMul.hSMul J N) (HSMul.hSMul (Neg.neg (HSub.hSub (HMul.hMul …
  -/
  exact Submodule.smul_mem_smul (Submodule.neg_mem _ hs) hn
  /-
    🎉 no goals
  -/


lemma eq_bot_of_eq_ideal_smul_of_le_jacobson_annihilator {I : Ideal R}
    {N : Submodule R M} (hN : FG N) (hIN : N = I • N)
    (hIjac : I ≤ N.annihilator.jacobson) : N = ⊥ :=
  (eq_smul_of_le_smul_of_le_jacobson hN hIN.le hIjac).trans N.annihilator_smul


open Pointwise in
lemma eq_bot_of_eq_pointwise_smul_of_mem_jacobson_annihilator {r : R}
    {N : Submodule R M} (hN : FG N) (hrN : N = r • N)
    (hrJac : r ∈ N.annihilator.jacobson) : N = ⊥ :=
  eq_bot_of_eq_ideal_smul_of_le_jacobson_annihilator hN
    (Eq.trans hrN (ideal_span_singleton_smul r N).symm)
    ((span_singleton_le_iff_mem r _).mpr hrJac)


open Pointwise in
lemma eq_bot_of_set_smul_eq_of_subset_jacobson_annihilator {s : Set R}
    {N : Submodule R M} (hN : FG N) (hsN : N = s • N)
    (hsJac : s ⊆ N.annihilator.jacobson) : N = ⊥ :=
  eq_bot_of_eq_ideal_smul_of_le_jacobson_annihilator hN
    (Eq.trans hsN (span_smul_eq s N).symm) (span_le.mpr hsJac)


lemma top_ne_ideal_smul_of_le_jacobson_annihilator [Nontrivial M]
    [Module.Finite R M] {I} (h : I ≤ (Module.annihilator R M).jacobson) :
    (⊤ : Submodule R M) ≠ I • ⊤ := fun H => top_ne_bot <|
  eq_bot_of_eq_ideal_smul_of_le_jacobson_annihilator Module.Finite.out H <|
    (congrArg (I ≤ Ideal.jacobson ·) annihilator_top).mpr h


open Pointwise in
lemma top_ne_set_smul_of_subset_jacobson_annihilator [Nontrivial M]
    [Module.Finite R M] {s : Set R}
    (h : s ⊆ (Module.annihilator R M).jacobson) :
    (⊤ : Submodule R M) ≠ s • ⊤ :=
  ne_of_ne_of_eq (top_ne_ideal_smul_of_le_jacobson_annihilator (span_le.mpr h))
    (span_smul_eq _ _)


open Pointwise in
lemma top_ne_pointwise_smul_of_mem_jacobson_annihilator [Nontrivial M]
    [Module.Finite R M] {r} (h : r ∈ (Module.annihilator R M).jacobson) :
    (⊤ : Submodule R M) ≠ r • ⊤ :=
  ne_of_ne_of_eq (top_ne_set_smul_of_subset_jacobson_annihilator <|
                    Set.singleton_subset_iff.mpr h) (singleton_set_smul ⊤ r)


/-- **Nakayama's Lemma** - Statement (2) in
[Stacks 00DV](https://stacks.math.columbia.edu/tag/00DV).
See also `eq_smul_of_le_smul_of_le_jacobson` for a generalisation
to the `jacobson` of any ideal -/
@[stacks 00DV "(2)"]
theorem eq_bot_of_le_smul_of_le_jacobson_bot (I : Ideal R) (N : Submodule R M) (hN : N.FG)
    (hIN : N ≤ I • N) (hIjac : I ≤ jacobson ⊥) : N = ⊥ := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    N : Submodule R M
    hN : N.FG
    hIN : LE.le N (HSMul.hSMul I N)
    hIjac : LE.le I Bot.bot.jacobson
    ⊢ Eq N Bot.bot
  -/
  rw [eq_smul_of_le_smul_of_le_jacobson hN hIN hIjac, Submodule.bot_smul]
  /-
    🎉 no goals
  -/


theorem sup_eq_sup_smul_of_le_smul_of_le_jacobson {I J : Ideal R} {N N' : Submodule R M}
    (hN' : N'.FG) (hIJ : I ≤ jacobson J) (hNN : N' ≤ N ⊔ I • N') : N ⊔ N' = N ⊔ J • N' := by
  have hNN' : N ⊔ N' = N ⊔ I • N' :=
    le_antisymm (sup_le le_sup_left hNN)
    (sup_le_sup_left (Submodule.smul_le.2 fun _ _ _ => Submodule.smul_mem _ _) _)
  have h_comap :=
    comap_injective_of_surjective (LinearMap.range_eq_top.1 N.range_mkQ)
  have : (I • N').map N.mkQ = N'.map N.mkQ := by
    simpa only [← h_comap.eq_iff, comap_map_mkQ, sup_comm, eq_comm] using hNN'
  have :=
    @Submodule.eq_smul_of_le_smul_of_le_jacobson _ _ _ _ _ I J (N'.map N.mkQ) (hN'.map _)
      (by rw [← map_smul'', this]) hIJ
  rwa [← map_smul'', ← h_comap.eq_iff, comap_map_eq, comap_map_eq, Submodule.ker_mkQ, sup_comm,
    sup_comm (b := N)] at this


/-- **Nakayama's Lemma** - A slightly more general version of (4) in
[Stacks 00DV](https://stacks.math.columbia.edu/tag/00DV).
See also `smul_le_of_le_smul_of_le_jacobson_bot` for the special case when `J = ⊥`. -/
@[stacks 00DV "(4)"]
theorem sup_smul_eq_sup_smul_of_le_smul_of_le_jacobson {I J : Ideal R} {N N' : Submodule R M}
    (hN' : N'.FG) (hIJ : I ≤ jacobson J) (hNN : N' ≤ N ⊔ I • N') : N ⊔ I • N' = N ⊔ J • N' :=
  ((sup_le_sup_left smul_le_right _).antisymm (sup_le le_sup_left hNN)).trans
    (sup_eq_sup_smul_of_le_smul_of_le_jacobson hN' hIJ hNN)


theorem le_of_le_smul_of_le_jacobson_bot {R M} [CommRing R] [AddCommGroup M] [Module R M]
    {I : Ideal R} {N N' : Submodule R M} (hN' : N'.FG)
    (hIJ : I ≤ jacobson ⊥) (hNN : N' ≤ N ⊔ I • N') : N' ≤ N := by
  /-
    R : Type u_3
    M : Type u_4
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    N N' : Submodule R M
    hN' : N'.FG
    hIJ : LE.le I Bot.bot.jacobson
    hNN : LE.le N' (Max.max N (HSMul.hSMul I N'))
    ⊢ LE.le N' N
  -/
  rw [← sup_eq_left, sup_eq_sup_smul_of_le_smul_of_le_jacobson hN' hIJ hNN, bot_smul, sup_bot_eq]
  /-
    🎉 no goals
  -/


/-- **Nakayama's Lemma** - Statement (4) in
[Stacks 00DV](https://stacks.math.columbia.edu/tag/00DV).
See also `sup_smul_eq_sup_smul_of_le_smul_of_le_jacobson` for a generalisation
to the `jacobson` of any ideal -/
@[stacks 00DV "(4)"]
theorem smul_le_of_le_smul_of_le_jacobson_bot {I : Ideal R} {N N' : Submodule R M} (hN' : N'.FG)
    (hIJ : I ≤ jacobson ⊥) (hNN : N' ≤ N ⊔ I • N') : I • N' ≤ N :=
  smul_le_right.trans (le_of_le_smul_of_le_jacobson_bot hN' hIJ hNN)


@[stacks 00DV "(3) see `Submodule.localized₀_le_localized₀_of_smul_le` for the second conclusion."]
lemma exists_sub_one_mem_and_smul_le_of_fg_of_le_sup {I : Ideal R}
    {N N' P : Submodule R M} (hN' : N'.FG) (hN'le : N' ≤ P) (hNN' : P ≤ N ⊔ I • N') :
    ∃ r : R, r - 1 ∈ I ∧ r • P ≤ N := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    N N' P : Submodule R M
    hN' : N'.FG
    hN'le : LE.le N' P
    hNN' : LE.le P (Max.max N (HSMul.hSMul I N'))
    ⊢ Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (LE.le (HSMul.hSMul r …
  -/
  have hNN'' : P ≤ N ⊔ N' := le_trans hNN' (by simpa using le_trans smul_le_right le_sup_right)
  have h1 : P.map N.mkQ = N'.map N.mkQ := by
    refine le_antisymm ?_ (map_mono hN'le)
    simpa using map_mono (f := N.mkQ) hNN''
  have h2 : P.map N.mkQ = (I • N').map N.mkQ := by
    apply le_antisymm
    · simpa using map_mono (f := N.mkQ) hNN'
    · rw [h1]
      simp [smul_le_right]
  have hle : (P.map N.mkQ) ≤ I • P.map N.mkQ := by
    conv_lhs => rw [h2]
    simp [← h1]
  obtain ⟨r, hmem, hr⟩ := exists_sub_one_mem_and_smul_eq_zero_of_fg_of_le_smul I _
    (h1 ▸ hN'.map _) hle
  /-
    case intro.intro
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    N N' P : Submodule R M
    hN' : N'.FG
    hN'le : LE.le N' P
    hNN' : LE.le P (Max.max N (HSMul.hSMul I N'))
    hNN'' : LE.le P (Max.max N N')
    h1 : Eq (Submodule.map N.mkQ P) (Submodule.map N.mkQ N')
    h2 : Eq (Submodule.map N.mkQ P) (Submodule.map N.mkQ (HSMul.hSMul I N'))
    hle : LE.le (Submodule.map N.mkQ P) (HSMul.hSMul I (Submodule.map N.mkQ P))
    r : R
    hmem : Membership.mem I (HSub.hSub r 1)
    hr : ∀ (n : HasQuotient.Quotient M N), Membership.mem (Submodule.map N.mkQ P)  …
    ⊢ Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (LE.le (HSMul.hSMul r …
  -/
  refine ⟨r, hmem, fun x hx ↦ ?_⟩
  /-
    case intro.intro
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    N N' P : Submodule R M
    hN' : N'.FG
    hN'le : LE.le N' P
    hNN' : LE.le P (Max.max N (HSMul.hSMul I N'))
    hNN'' : LE.le P (Max.max N N')
    h1 : Eq (Submodule.map N.mkQ P) (Submodule.map N.mkQ N')
    h2 : Eq (Submodule.map N.mkQ P) (Submodule.map N.mkQ (HSMul.hSMul I N'))
    hle : LE.le (Submodule.map N.mkQ P) (HSMul.hSMul I (Submodule.map N.mkQ P))
    r : R
    hmem : Membership.mem I (HSub.hSub r 1)
    hr : ∀ (n : HasQuotient.Quotient M N), Membership.mem (Submodule.map N.mkQ P)  …
    x : M
    hx : Membership.mem (HSMul.hSMul r P) x
    ⊢ Membership.mem N x
  -/
  induction' hx using Submodule.smul_inductionOn_pointwise with p hp _ _ _ h _ _ _ _ hx hy
    /-
      case intro.intro.smul₀
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      N N' P : Submodule R M
      hN' : N'.FG
      hN'le : LE.le N' P
      hNN' : LE.le P (Max.max N (HSMul.hSMul I N'))
      hNN'' : LE.le P (Max.max N N')
      h1 : Eq (Submodule.map N.mkQ P) (Submodule.map N.mkQ N')
      h2 : Eq (Submodule.map N.mkQ P) (Submodule.map N.mkQ (HSMul.hSMul I N'))
      hle : LE.le (Submodule.map N.mkQ P) (HSMul.hSMul I (Submodule.map N.mkQ P))
      r : R
      hmem : Membership.mem I (HSub.hSub r 1)
      hr : ∀ (n : HasQuotient.Quotient M N), Membership.mem (Submodule.map N.mkQ P)  …
      x p : M
      hp : Membership.mem P p
      ⊢ Membership.mem N (HSMul.hSMul r p)
    -/
  · rw [← Submodule.Quotient.mk_eq_zero, Quotient.mk_smul]
    /-
      case intro.intro.smul₀
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      N N' P : Submodule R M
      hN' : N'.FG
      hN'le : LE.le N' P
      hNN' : LE.le P (Max.max N (HSMul.hSMul I N'))
      hNN'' : LE.le P (Max.max N N')
      h1 : Eq (Submodule.map N.mkQ P) (Submodule.map N.mkQ N')
      h2 : Eq (Submodule.map N.mkQ P) (Submodule.map N.mkQ (HSMul.hSMul I N'))
      hle : LE.le (Submodule.map N.mkQ P) (HSMul.hSMul I (Submodule.map N.mkQ P))
      r : R
      hmem : Membership.mem I (HSub.hSub r 1)
      hr : ∀ (n : HasQuotient.Quotient M N), Membership.mem (Submodule.map N.mkQ P)  …
      x p : M
      hp : Membership.mem P p
      ⊢ Eq (HSMul.hSMul r (Submodule.Quotient.mk p)) 0
    -/
    exact hr _ ⟨p, hp, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.smul₁
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      N N' P : Submodule R M
      hN' : N'.FG
      hN'le : LE.le N' P
      hNN' : LE.le P (Max.max N (HSMul.hSMul I N'))
      hNN'' : LE.le P (Max.max N N')
      h1 : Eq (Submodule.map N.mkQ P) (Submodule.map N.mkQ N')
      h2 : Eq (Submodule.map N.mkQ P) (Submodule.map N.mkQ (HSMul.hSMul I N'))
      hle : LE.le (Submodule.map N.mkQ P) (HSMul.hSMul I (Submodule.map N.mkQ P))
      r : R
      hmem : Membership.mem I (HSub.hSub r 1)
      hr : ∀ (n : HasQuotient.Quotient M N), Membership.mem (Submodule.map N.mkQ P)  …
      x : M
      r✝ : R
      m✝ : M
      mem✝ : Membership.mem (HSMul.hSMul r P) m✝
      h : Membership.mem N m✝
      ⊢ Membership.mem N (HSMul.hSMul r✝ m✝)
    -/
  · exact N.smul_mem _ h
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.add
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      N N' P : Submodule R M
      hN' : N'.FG
      hN'le : LE.le N' P
      hNN' : LE.le P (Max.max N (HSMul.hSMul I N'))
      hNN'' : LE.le P (Max.max N N')
      h1 : Eq (Submodule.map N.mkQ P) (Submodule.map N.mkQ N')
      h2 : Eq (Submodule.map N.mkQ P) (Submodule.map N.mkQ (HSMul.hSMul I N'))
      hle : LE.le (Submodule.map N.mkQ P) (HSMul.hSMul I (Submodule.map N.mkQ P))
      r : R
      hmem : Membership.mem I (HSub.hSub r 1)
      hr : ∀ (n : HasQuotient.Quotient M N), Membership.mem (Submodule.map N.mkQ P)  …
      x x✝ y✝ : M
      hx✝ : Membership.mem (HSMul.hSMul r P) x✝
      hy✝ : Membership.mem (HSMul.hSMul r P) y✝
      hx : Membership.mem N x✝
      hy : Membership.mem N y✝
      ⊢ Membership.mem N (HAdd.hAdd x✝ y✝)
    -/
  · exact N.add_mem hx hy
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.zero
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      N N' P : Submodule R M
      hN' : N'.FG
      hN'le : LE.le N' P
      hNN' : LE.le P (Max.max N (HSMul.hSMul I N'))
      hNN'' : LE.le P (Max.max N N')
      h1 : Eq (Submodule.map N.mkQ P) (Submodule.map N.mkQ N')
      h2 : Eq (Submodule.map N.mkQ P) (Submodule.map N.mkQ (HSMul.hSMul I N'))
      hle : LE.le (Submodule.map N.mkQ P) (HSMul.hSMul I (Submodule.map N.mkQ P))
      r : R
      hmem : Membership.mem I (HSub.hSub r 1)
      hr : ∀ (n : HasQuotient.Quotient M N), Membership.mem (Submodule.map N.mkQ P)  …
      x : M
      ⊢ Membership.mem N 0
    -/
  · exact N.zero_mem
    /-
      🎉 no goals
    -/


