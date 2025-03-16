theorem exists_smul_add_of_span_sup_eq_top (y : L) : ∃ t : R, ∃ z ∈ I, y = t • x + z := by
  /-
    R : Type u₁
    L : Type u₂
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    x : L
    hxI : Eq (Max.max (Submodule.span R (Singleton.singleton x)) (LieIdeal.toLieSu …
    y : L
    ⊢ Exists fun t => Exists fun z => And (Membership.mem I z) (Eq y (HAdd.hAdd (H …
  -/
  have hy : y ∈ (⊤ : Submodule R L) := Submodule.mem_top
  /-
    R : Type u₁
    L : Type u₂
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    x : L
    hxI : Eq (Max.max (Submodule.span R (Singleton.singleton x)) (LieIdeal.toLieSu …
    y : L
    hy : Membership.mem Top.top y
    ⊢ Exists fun t => Exists fun z => And (Membership.mem I z) (Eq y (HAdd.hAdd (H …
  -/
  simp only [← hxI, Submodule.mem_sup, Submodule.mem_span_singleton] at hy
  /-
    R : Type u₁
    L : Type u₂
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    x : L
    hxI : Eq (Max.max (Submodule.span R (Singleton.singleton x)) (LieIdeal.toLieSu …
    y : L
    hy : Exists fun y_1 => And (Exists fun a => Eq (HSMul.hSMul a x) y_1) (Exists  …
    ⊢ Exists fun t => Exists fun z => And (Membership.mem I z) (Eq y (HAdd.hAdd (H …
  -/
  obtain ⟨-, ⟨t, rfl⟩, z, hz, rfl⟩ := hy
  /-
    case intro.intro.intro.intro.intro
    R : Type u₁
    L : Type u₂
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    x : L
    hxI : Eq (Max.max (Submodule.span R (Singleton.singleton x)) (LieIdeal.toLieSu …
    t : R
    z : L
    hz : Membership.mem (LieIdeal.toLieSubalgebra R L I).toSubmodule z
    ⊢ Exists fun t_1 => Exists fun z_1 => And (Membership.mem I z_1) (Eq (HAdd.hAd …
  -/
  exact ⟨t, z, hz, rfl⟩
  /-
    🎉 no goals
  -/


theorem lie_top_eq_of_span_sup_eq_top (N : LieSubmodule R L M) :
    (↑⁅(⊤ : LieIdeal R L), N⁆ : Submodule R M) =
      (N : Submodule R M).map (toEnd R L M x) ⊔ (↑⁅I, N⁆ : Submodule R M) := by
  simp only [lieIdeal_oper_eq_linear_span', Submodule.sup_span, mem_top, exists_prop,
    true_and, Submodule.map_coe, toEnd_apply_apply]
  /-
    R : Type u₁
    L : Type u₂
    M : Type u₄
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    I : LieIdeal R L
    x : L
    hxI : Eq (Max.max (Submodule.span R (Singleton.singleton x)) (LieIdeal.toLieSu …
    N : LieSubmodule R L M
    ⊢ Eq (Submodule.span R (setOf fun m => Exists fun x => Exists fun n => And (Me …
  -/
  refine le_antisymm (Submodule.span_le.mpr ?_) (Submodule.span_mono fun z hz => ?_)
    /-
      case refine_1
      R : Type u₁
      L : Type u₂
      M : Type u₄
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      I : LieIdeal R L
      x : L
      hxI : Eq (Max.max (Submodule.span R (Singleton.singleton x)) (LieIdeal.toLieSu …
      N : LieSubmodule R L M
      ⊢ HasSubset.Subset (setOf fun m => Exists fun x => Exists fun n => And (Member …
    -/
  · rintro z ⟨y, n, hn : n ∈ N, rfl⟩
    /-
      case refine_1.intro.intro.intro
      R : Type u₁
      L : Type u₂
      M : Type u₄
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      I : LieIdeal R L
      x : L
      hxI : Eq (Max.max (Submodule.span R (Singleton.singleton x)) (LieIdeal.toLieSu …
      N : LieSubmodule R L M
      y : L
      n : M
      hn : Membership.mem N n
      ⊢ Membership.mem (↑(Submodule.span R (Union.union (Set.image (fun a => Bracket …
    -/
    obtain ⟨t, z, hz, rfl⟩ := exists_smul_add_of_span_sup_eq_top hxI y
    /-
      case refine_1.intro.intro.intro.intro.intro.intro
      R : Type u₁
      L : Type u₂
      M : Type u₄
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      I : LieIdeal R L
      x : L
      hxI : Eq (Max.max (Submodule.span R (Singleton.singleton x)) (LieIdeal.toLieSu …
      N : LieSubmodule R L M
      n : M
      hn : Membership.mem N n
      t : R
      z : L
      hz : Membership.mem I z
      ⊢ Membership.mem (↑(Submodule.span R (Union.union (Set.image (fun a => Bracket …
    -/
    simp only [SetLike.mem_coe, Submodule.span_union, Submodule.mem_sup]
    exact
      ⟨t • ⁅x, n⁆, Submodule.subset_span ⟨t • n, N.smul_mem' t hn, lie_smul t x n⟩, ⁅z, n⁆,
        Submodule.subset_span ⟨z, hz, n, hn, rfl⟩, by simp⟩
    /-
      case refine_2
      R : Type u₁
      L : Type u₂
      M : Type u₄
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      I : LieIdeal R L
      x : L
      hxI : Eq (Max.max (Submodule.span R (Singleton.singleton x)) (LieIdeal.toLieSu …
      N : LieSubmodule R L M
      z : M
      hz : Membership.mem (Union.union (Set.image (fun a => Bracket.bracket x a) ↑↑N …
      ⊢ Membership.mem (setOf fun m => Exists fun x => Exists fun n => And (Membersh …
    -/
  · rcases hz with (⟨m, hm, rfl⟩ | ⟨y, -, m, hm, rfl⟩)
    /-
      case refine_2.inl.intro.intro
      R : Type u₁
      L : Type u₂
      M : Type u₄
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      I : LieIdeal R L
      x : L
      hxI : Eq (Max.max (Submodule.span R (Singleton.singleton x)) (LieIdeal.toLieSu …
      N : LieSubmodule R L M
      m : M
      hm : Membership.mem (↑↑N) m
      ⊢ Membership.mem (setOf fun m => Exists fun x => Exists fun n => And (Membersh …
    -/
    exacts [⟨x, m, hm, rfl⟩, ⟨y, m, hm, rfl⟩]
    /-
      🎉 no goals
    -/


theorem lcs_le_lcs_of_is_nilpotent_span_sup_eq_top {n i j : ℕ}
    (hxn : toEnd R L M x ^ n = 0) (hIM : lowerCentralSeries R L M i ≤ I.lcs M j) :
    lowerCentralSeries R L M (i + n) ≤ I.lcs M (j + 1) := by
  suffices
    ∀ l,
      ((⊤ : LieIdeal R L).lcs M (i + l) : Submodule R M) ≤
        (I.lcs M j : Submodule R M).map (toEnd R L M x ^ l) ⊔
          (I.lcs M (j + 1) : Submodule R M)
    by simpa only [bot_sup_eq, LieIdeal.incl_coe, Submodule.map_zero, hxn] using this n
  /-
    R : Type u₁
    L : Type u₂
    M : Type u₄
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    I : LieIdeal R L
    x : L
    hxI : Eq (Max.max (Submodule.span R (Singleton.singleton x)) (LieIdeal.toLieSu …
    n i j : Nat
    hxn : Eq (HPow.hPow ((LieModule.toEnd R L M) x) n) 0
    hIM : LE.le (LieModule.lowerCentralSeries R L M i) (I.lcs M j)
    ⊢ ∀ (l : Nat), LE.le (↑(Top.top.lcs M (HAdd.hAdd i l))) (Max.max (Submodule.ma …
  -/
  intro l
  induction l with
  | zero =>
    simp only [add_zero, LieIdeal.lcs_succ, pow_zero, LinearMap.one_eq_id,
      Submodule.map_id]
    exact le_sup_of_le_left hIM
  | succ l ih =>
    simp only [LieIdeal.lcs_succ, i.add_succ l, lie_top_eq_of_span_sup_eq_top hxI, sup_le_iff]
    refine ⟨(Submodule.map_mono ih).trans ?_, le_sup_of_le_right ?_⟩
    · rw [Submodule.map_sup, ← Submodule.map_comp, ← LinearMap.mul_eq_comp, ← pow_succ', ←
        I.lcs_succ]
      exact sup_le_sup_left coe_map_toEnd_le _
    · refine le_trans (mono_lie_right I ?_) (mono_lie_right I hIM)
      exact antitone_lowerCentralSeries R L M le_self_add


theorem isNilpotentOfIsNilpotentSpanSupEqTop (hnp : IsNilpotent <| toEnd R L M x)
    (hIM : IsNilpotent R I M) : IsNilpotent R L M := by
  /-
    R : Type u₁
    L : Type u₂
    M : Type u₄
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    I : LieIdeal R L
    x : L
    hxI : Eq (Max.max (Submodule.span R (Singleton.singleton x)) (LieIdeal.toLieSu …
    hnp : IsNilpotent ((LieModule.toEnd R L M) x)
    hIM : LieModule.IsNilpotent R (Subtype fun x => Membership.mem I x) M
    ⊢ LieModule.IsNilpotent R L M
  -/
  obtain ⟨n, hn⟩ := hnp
  /-
    case intro
    R : Type u₁
    L : Type u₂
    M : Type u₄
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    I : LieIdeal R L
    x : L
    hxI : Eq (Max.max (Submodule.span R (Singleton.singleton x)) (LieIdeal.toLieSu …
    hIM : LieModule.IsNilpotent R (Subtype fun x => Membership.mem I x) M
    n : Nat
    hn : Eq (HPow.hPow ((LieModule.toEnd R L M) x) n) 0
    ⊢ LieModule.IsNilpotent R L M
  -/
  obtain ⟨k, hk⟩ := hIM
  have hk' : I.lcs M k = ⊥ := by
    simp only [← toSubmodule_inj, I.coe_lcs_eq, hk, bot_toSubmodule]
  suffices ∀ l, lowerCentralSeries R L M (l * n) ≤ I.lcs M l by
    use k * n
    simpa [hk'] using this k
  /-
    case intro.mk.intro
    R : Type u₁
    L : Type u₂
    M : Type u₄
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    I : LieIdeal R L
    x : L
    hxI : Eq (Max.max (Submodule.span R (Singleton.singleton x)) (LieIdeal.toLieSu …
    n : Nat
    hn : Eq (HPow.hPow ((LieModule.toEnd R L M) x) n) 0
    k : Nat
    hk : Eq (LieModule.lowerCentralSeries R (Subtype fun x => Membership.mem I x)  …
    hk' : Eq (I.lcs M k) Bot.bot
    ⊢ ∀ (l : Nat), LE.le (LieModule.lowerCentralSeries R L M (HMul.hMul l n)) (I.l …
  -/
  intro l
  induction l with
  | zero => simp
  | succ l ih => exact (l.succ_mul n).symm ▸ lcs_le_lcs_of_is_nilpotent_span_sup_eq_top hxI hn ih


/-- A Lie algebra `L` is said to be Engelian if a sufficient condition for any `L`-Lie module `M` to
be nilpotent is that the image of the map `L → End(M)` consists of nilpotent elements.

Engel's theorem `LieAlgebra.isEngelian_of_isNoetherian` states that any Noetherian Lie algebra is
Engelian. -/
def LieAlgebra.IsEngelian : Prop :=
  ∀ (M : Type u₄) [AddCommGroup M] [Module R M] [LieRingModule L M] [LieModule R L M],
    (∀ x : L, _root_.IsNilpotent (toEnd R L M x)) → LieModule.IsNilpotent R L M


theorem LieAlgebra.isEngelian_of_subsingleton [Subsingleton L] : LieAlgebra.IsEngelian R L := by
  /-
    R : Type u₁
    L : Type u₂
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : Subsingleton L
    ⊢ LieAlgebra.IsEngelian R L
  -/
  intro M _i1 _i2 _i3 _i4 _h
  /-
    R : Type u₁
    L : Type u₂
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : Subsingleton L
    M : Type u_1
    _i1 : AddCommGroup M
    _i2 : Module R M
    _i3 : LieRingModule L M
    _i4 : LieModule R L M
    _h : ∀ (x : L), _root_.IsNilpotent ((LieModule.toEnd R L M) x)
    ⊢ LieModule.IsNilpotent R L M
  -/
  use 1
  /-
    case h
    R : Type u₁
    L : Type u₂
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : Subsingleton L
    M : Type u_1
    _i1 : AddCommGroup M
    _i2 : Module R M
    _i3 : LieRingModule L M
    _i4 : LieModule R L M
    _h : ∀ (x : L), _root_.IsNilpotent ((LieModule.toEnd R L M) x)
    ⊢ Eq (LieModule.lowerCentralSeries R L M 1) Bot.bot
  -/
  suffices (⊤ : LieIdeal R L) = ⊥ by simp [this]
  /-
    case h
    R : Type u₁
    L : Type u₂
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : Subsingleton L
    M : Type u_1
    _i1 : AddCommGroup M
    _i2 : Module R M
    _i3 : LieRingModule L M
    _i4 : LieModule R L M
    _h : ∀ (x : L), _root_.IsNilpotent ((LieModule.toEnd R L M) x)
    ⊢ Eq Top.top Bot.bot
  -/
  subsingleton [(LieSubmodule.subsingleton_iff R L L).mpr inferInstance]
  /-
    🎉 no goals
  -/


theorem Function.Surjective.isEngelian {f : L →ₗ⁅R⁆ L₂} (hf : Function.Surjective f)
    (h : LieAlgebra.IsEngelian.{u₁, u₂, u₄} R L) : LieAlgebra.IsEngelian.{u₁, u₃, u₄} R L₂ := by
  /-
    R : Type u₁
    L : Type u₂
    L₂ : Type u₃
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L₂
    inst✝ : LieAlgebra R L₂
    f : LieHom R L L₂
    hf : Function.Surjective ⇑f
    h : LieAlgebra.IsEngelian R L
    ⊢ LieAlgebra.IsEngelian R L₂
  -/
  intro M _i1 _i2 _i3 _i4 h'
  /-
    R : Type u₁
    L : Type u₂
    L₂ : Type u₃
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L₂
    inst✝ : LieAlgebra R L₂
    f : LieHom R L L₂
    hf : Function.Surjective ⇑f
    h : LieAlgebra.IsEngelian R L
    M : Type u₄
    _i1 : AddCommGroup M
    _i2 : Module R M
    _i3 : LieRingModule L₂ M
    _i4 : LieModule R L₂ M
    h' : ∀ (x : L₂), IsNilpotent ((LieModule.toEnd R L₂ M) x)
    ⊢ LieModule.IsNilpotent R L₂ M
  -/
  letI : LieRingModule L M := LieRingModule.compLieHom M f
  /-
    R : Type u₁
    L : Type u₂
    L₂ : Type u₃
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L₂
    inst✝ : LieAlgebra R L₂
    f : LieHom R L L₂
    hf : Function.Surjective ⇑f
    h : LieAlgebra.IsEngelian R L
    M : Type u₄
    _i1 : AddCommGroup M
    _i2 : Module R M
    _i3 : LieRingModule L₂ M
    _i4 : LieModule R L₂ M
    h' : ∀ (x : L₂), IsNilpotent ((LieModule.toEnd R L₂ M) x)
    this : LieRingModule L M := LieRingModule.compLieHom M f
    ⊢ LieModule.IsNilpotent R L₂ M
  -/
  letI : LieModule R L M := compLieHom M f
  /-
    R : Type u₁
    L : Type u₂
    L₂ : Type u₃
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L₂
    inst✝ : LieAlgebra R L₂
    f : LieHom R L L₂
    hf : Function.Surjective ⇑f
    h : LieAlgebra.IsEngelian R L
    M : Type u₄
    _i1 : AddCommGroup M
    _i2 : Module R M
    _i3 : LieRingModule L₂ M
    _i4 : LieModule R L₂ M
    h' : ∀ (x : L₂), IsNilpotent ((LieModule.toEnd R L₂ M) x)
    this✝ : LieRingModule L M := LieRingModule.compLieHom M f
    this : LieModule R L M := LieModule.compLieHom M f
    ⊢ LieModule.IsNilpotent R L₂ M
  -/
  have hnp : ∀ x, IsNilpotent (toEnd R L M x) := fun x => h' (f x)
  /-
    R : Type u₁
    L : Type u₂
    L₂ : Type u₃
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L₂
    inst✝ : LieAlgebra R L₂
    f : LieHom R L L₂
    hf : Function.Surjective ⇑f
    h : LieAlgebra.IsEngelian R L
    M : Type u₄
    _i1 : AddCommGroup M
    _i2 : Module R M
    _i3 : LieRingModule L₂ M
    _i4 : LieModule R L₂ M
    h' : ∀ (x : L₂), IsNilpotent ((LieModule.toEnd R L₂ M) x)
    this✝ : LieRingModule L M := LieRingModule.compLieHom M f
    this : LieModule R L M := LieModule.compLieHom M f
    hnp : ∀ (x : L), IsNilpotent ((LieModule.toEnd R L M) x)
    ⊢ LieModule.IsNilpotent R L₂ M
  -/
  have surj_id : Function.Surjective (LinearMap.id : M →ₗ[R] M) := Function.surjective_id
  /-
    R : Type u₁
    L : Type u₂
    L₂ : Type u₃
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L₂
    inst✝ : LieAlgebra R L₂
    f : LieHom R L L₂
    hf : Function.Surjective ⇑f
    h : LieAlgebra.IsEngelian R L
    M : Type u₄
    _i1 : AddCommGroup M
    _i2 : Module R M
    _i3 : LieRingModule L₂ M
    _i4 : LieModule R L₂ M
    h' : ∀ (x : L₂), IsNilpotent ((LieModule.toEnd R L₂ M) x)
    this✝ : LieRingModule L M := LieRingModule.compLieHom M f
    this : LieModule R L M := LieModule.compLieHom M f
    hnp : ∀ (x : L), IsNilpotent ((LieModule.toEnd R L M) x)
    surj_id : Function.Surjective ⇑LinearMap.id
    ⊢ LieModule.IsNilpotent R L₂ M
  -/
  haveI : LieModule.IsNilpotent R L M := h M hnp
  /-
    R : Type u₁
    L : Type u₂
    L₂ : Type u₃
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L₂
    inst✝ : LieAlgebra R L₂
    f : LieHom R L L₂
    hf : Function.Surjective ⇑f
    h : LieAlgebra.IsEngelian R L
    M : Type u₄
    _i1 : AddCommGroup M
    _i2 : Module R M
    _i3 : LieRingModule L₂ M
    _i4 : LieModule R L₂ M
    h' : ∀ (x : L₂), IsNilpotent ((LieModule.toEnd R L₂ M) x)
    this✝¹ : LieRingModule L M := LieRingModule.compLieHom M f
    this✝ : LieModule R L M := LieModule.compLieHom M f
    hnp : ∀ (x : L), IsNilpotent ((LieModule.toEnd R L M) x)
    surj_id : Function.Surjective ⇑LinearMap.id
    this : LieModule.IsNilpotent R L M
    ⊢ LieModule.IsNilpotent R L₂ M
  -/
  apply hf.lieModuleIsNilpotent surj_id
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10745): was `simp`
  /-
    case hfg
    R : Type u₁
    L : Type u₂
    L₂ : Type u₃
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L₂
    inst✝ : LieAlgebra R L₂
    f : LieHom R L L₂
    hf : Function.Surjective ⇑f
    h : LieAlgebra.IsEngelian R L
    M : Type u₄
    _i1 : AddCommGroup M
    _i2 : Module R M
    _i3 : LieRingModule L₂ M
    _i4 : LieModule R L₂ M
    h' : ∀ (x : L₂), IsNilpotent ((LieModule.toEnd R L₂ M) x)
    this✝¹ : LieRingModule L M := LieRingModule.compLieHom M f
    this✝ : LieModule R L M := LieModule.compLieHom M f
    hnp : ∀ (x : L), IsNilpotent ((LieModule.toEnd R L M) x)
    surj_id : Function.Surjective ⇑LinearMap.id
    this : LieModule.IsNilpotent R L M
    ⊢ ∀ (x : L) (m : M), Eq (Bracket.bracket (f x) (LinearMap.id m)) (LinearMap.id …
  -/
  intros; simp only [LinearMap.id_coe, id_eq]; rfl
                                               /-
                                                 🎉 no goals
                                               -/


theorem LieEquiv.isEngelian_iff (e : L ≃ₗ⁅R⁆ L₂) :
    LieAlgebra.IsEngelian.{u₁, u₂, u₄} R L ↔ LieAlgebra.IsEngelian.{u₁, u₃, u₄} R L₂ :=
  ⟨e.surjective.isEngelian, e.symm.surjective.isEngelian⟩

-- Porting note: changed statement from `∃ ∃ ..` to `∃ .. ∧ ..`

theorem LieAlgebra.exists_engelian_lieSubalgebra_of_lt_normalizer {K : LieSubalgebra R L}
    (hK₁ : LieAlgebra.IsEngelian.{u₁, u₂, u₄} R K) (hK₂ : K < K.normalizer) :
    ∃ (K' : LieSubalgebra R L), LieAlgebra.IsEngelian.{u₁, u₂, u₄} R K' ∧ K < K' := by
  /-
    R : Type u₁
    L : Type u₂
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    K : LieSubalgebra R L
    hK₁ : LieAlgebra.IsEngelian R (Subtype fun x => Membership.mem K x)
    hK₂ : LT.lt K K.normalizer
    ⊢ Exists fun K' => And (LieAlgebra.IsEngelian R (Subtype fun x => Membership.m …
  -/
  obtain ⟨x, hx₁, hx₂⟩ := SetLike.exists_of_lt hK₂
  let K' : LieSubalgebra R L :=
    { (R ∙ x) ⊔ (K : Submodule R L) with
      lie_mem' := fun {y z} => LieSubalgebra.lie_mem_sup_of_mem_normalizer hx₁ }
  /-
    case intro.intro
    R : Type u₁
    L : Type u₂
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    K : LieSubalgebra R L
    hK₁ : LieAlgebra.IsEngelian R (Subtype fun x => Membership.mem K x)
    hK₂ : LT.lt K K.normalizer
    x : L
    hx₁ : Membership.mem K.normalizer x
    hx₂ : Not (Membership.mem K x)
    K' : LieSubalgebra R L :=
      let __src := Max.max (Submodule.span R (Singleton.singleton x)) K.toSubmodule;
      { toSubmodule := __src, lie_mem' := ⋯ }
    ⊢ Exists fun K' => And (LieAlgebra.IsEngelian R (Subtype fun x => Membership.m …
  -/
  have hxK' : x ∈ K' := Submodule.mem_sup_left (Submodule.subset_span (Set.mem_singleton _))
  /-
    case intro.intro
    R : Type u₁
    L : Type u₂
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    K : LieSubalgebra R L
    hK₁ : LieAlgebra.IsEngelian R (Subtype fun x => Membership.mem K x)
    hK₂ : LT.lt K K.normalizer
    x : L
    hx₁ : Membership.mem K.normalizer x
    hx₂ : Not (Membership.mem K x)
    K' : LieSubalgebra R L :=
      let __src := Max.max (Submodule.span R (Singleton.singleton x)) K.toSubmodule;
      { toSubmodule := __src, lie_mem' := ⋯ }
    hxK' : Membership.mem K' x
    ⊢ Exists fun K' => And (LieAlgebra.IsEngelian R (Subtype fun x => Membership.m …
  -/
  have hKK' : K ≤ K' := (LieSubalgebra.toSubmodule_le_toSubmodule K K').mp le_sup_right
  have hK' : K' ≤ K.normalizer := by
    rw [← LieSubalgebra.toSubmodule_le_toSubmodule]
    exact sup_le ((Submodule.span_singleton_le_iff_mem _ _).mpr hx₁) hK₂.le
  /-
    case intro.intro
    R : Type u₁
    L : Type u₂
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    K : LieSubalgebra R L
    hK₁ : LieAlgebra.IsEngelian R (Subtype fun x => Membership.mem K x)
    hK₂ : LT.lt K K.normalizer
    x : L
    hx₁ : Membership.mem K.normalizer x
    hx₂ : Not (Membership.mem K x)
    K' : LieSubalgebra R L :=
      let __src := Max.max (Submodule.span R (Singleton.singleton x)) K.toSubmodule;
      { toSubmodule := __src, lie_mem' := ⋯ }
    hxK' : Membership.mem K' x
    hKK' : LE.le K K'
    hK' : LE.le K' K.normalizer
    ⊢ Exists fun K' => And (LieAlgebra.IsEngelian R (Subtype fun x => Membership.m …
  -/
  refine ⟨K', ?_, lt_iff_le_and_ne.mpr ⟨hKK', fun contra => hx₂ (contra.symm ▸ hxK')⟩⟩
  /-
    case intro.intro
    R : Type u₁
    L : Type u₂
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    K : LieSubalgebra R L
    hK₁ : LieAlgebra.IsEngelian R (Subtype fun x => Membership.mem K x)
    hK₂ : LT.lt K K.normalizer
    x : L
    hx₁ : Membership.mem K.normalizer x
    hx₂ : Not (Membership.mem K x)
    K' : LieSubalgebra R L :=
      let __src := Max.max (Submodule.span R (Singleton.singleton x)) K.toSubmodule;
      { toSubmodule := __src, lie_mem' := ⋯ }
    hxK' : Membership.mem K' x
    hKK' : LE.le K K'
    hK' : LE.le K' K.normalizer
    ⊢ LieAlgebra.IsEngelian R (Subtype fun x => Membership.mem K' x)
  -/
  intro M _i1 _i2 _i3 _i4 h
  obtain ⟨I, hI₁ : (I : LieSubalgebra R K') = LieSubalgebra.ofLe hKK'⟩ :=
    LieSubalgebra.exists_nested_lieIdeal_ofLe_normalizer hKK' hK'
  have hI₂ : (R ∙ (⟨x, hxK'⟩ : K')) ⊔ (LieSubmodule.toSubmodule I) = ⊤ := by
    rw [← LieIdeal.toLieSubalgebra_toSubmodule R K' I, hI₁]
    apply Submodule.map_injective_of_injective (K' : Submodule R L).injective_subtype
    simp only [LieSubalgebra.coe_ofLe, Submodule.map_sup, Submodule.map_subtype_range_inclusion,
      Submodule.map_top, Submodule.range_subtype]
    rw [Submodule.map_subtype_span_singleton]
  have e : K ≃ₗ⁅R⁆ I :=
    (LieSubalgebra.equivOfLe hKK').trans
      (LieEquiv.ofEq _ _ ((LieSubalgebra.coe_set_eq _ _).mpr hI₁.symm))
  /-
    case intro.intro.intro
    R : Type u₁
    L : Type u₂
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    K : LieSubalgebra R L
    hK₁ : LieAlgebra.IsEngelian R (Subtype fun x => Membership.mem K x)
    hK₂ : LT.lt K K.normalizer
    x : L
    hx₁ : Membership.mem K.normalizer x
    hx₂ : Not (Membership.mem K x)
    K' : LieSubalgebra R L :=
      let __src := Max.max (Submodule.span R (Singleton.singleton x)) K.toSubmodule;
      { toSubmodule := __src, lie_mem' := ⋯ }
    hxK' : Membership.mem K' x
    hKK' : LE.le K K'
    hK' : LE.le K' K.normalizer
    M : Type u₄
    _i1 : AddCommGroup M
    _i2 : Module R M
    _i3 : LieRingModule (Subtype fun x => Membership.mem K' x) M
    _i4 : LieModule R (Subtype fun x => Membership.mem K' x) M
    h : ∀ (x : Subtype fun x => Membership.mem K' x), _root_.IsNilpotent ((LieModu …
    I : LieIdeal R (Subtype fun x => Membership.mem K' x)
    hI₁ : Eq (LieIdeal.toLieSubalgebra R (Subtype fun x => Membership.mem K' x) I) …
    hI₂ : Eq (Max.max (Submodule.span R (Singleton.singleton ⟨x, hxK'⟩)) ↑I) Top.top
    e : LieEquiv R (Subtype fun x => Membership.mem K x) (Subtype fun x => Members …
    ⊢ LieModule.IsNilpotent R (Subtype fun x => Membership.mem K' x) M
  -/
  have hI₃ : LieAlgebra.IsEngelian R I := e.isEngelian_iff.mp hK₁
  /-
    case intro.intro.intro
    R : Type u₁
    L : Type u₂
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    K : LieSubalgebra R L
    hK₁ : LieAlgebra.IsEngelian R (Subtype fun x => Membership.mem K x)
    hK₂ : LT.lt K K.normalizer
    x : L
    hx₁ : Membership.mem K.normalizer x
    hx₂ : Not (Membership.mem K x)
    K' : LieSubalgebra R L :=
      let __src := Max.max (Submodule.span R (Singleton.singleton x)) K.toSubmodule;
      { toSubmodule := __src, lie_mem' := ⋯ }
    hxK' : Membership.mem K' x
    hKK' : LE.le K K'
    hK' : LE.le K' K.normalizer
    M : Type u₄
    _i1 : AddCommGroup M
    _i2 : Module R M
    _i3 : LieRingModule (Subtype fun x => Membership.mem K' x) M
    _i4 : LieModule R (Subtype fun x => Membership.mem K' x) M
    h : ∀ (x : Subtype fun x => Membership.mem K' x), _root_.IsNilpotent ((LieModu …
    I : LieIdeal R (Subtype fun x => Membership.mem K' x)
    hI₁ : Eq (LieIdeal.toLieSubalgebra R (Subtype fun x => Membership.mem K' x) I) …
    hI₂ : Eq (Max.max (Submodule.span R (Singleton.singleton ⟨x, hxK'⟩)) ↑I) Top.top
    e : LieEquiv R (Subtype fun x => Membership.mem K x) (Subtype fun x => Members …
    hI₃ : LieAlgebra.IsEngelian R (Subtype fun x => Membership.mem I x)
    ⊢ LieModule.IsNilpotent R (Subtype fun x => Membership.mem K' x) M
  -/
  exact LieSubmodule.isNilpotentOfIsNilpotentSpanSupEqTop hI₂ (h _) (hI₃ _ fun x => h x)
  /-
    🎉 no goals
  -/


/-- *Engel's theorem*.

Note that this implies all traditional forms of Engel's theorem via
`LieModule.nontrivial_max_triv_of_isNilpotent`, `LieModule.isNilpotent_iff_forall`,
`LieAlgebra.isNilpotent_iff_forall`. -/
theorem LieAlgebra.isEngelian_of_isNoetherian [IsNoetherian R L] : LieAlgebra.IsEngelian R L := by
  /-
    R : Type u₁
    L : Type u₂
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : IsNoetherian R L
    ⊢ LieAlgebra.IsEngelian R L
  -/
  intro M _i1 _i2 _i3 _i4 h
  /-
    R : Type u₁
    L : Type u₂
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : IsNoetherian R L
    M : Type u_1
    _i1 : AddCommGroup M
    _i2 : Module R M
    _i3 : LieRingModule L M
    _i4 : LieModule R L M
    h : ∀ (x : L), _root_.IsNilpotent ((LieModule.toEnd R L M) x)
    ⊢ LieModule.IsNilpotent R L M
  -/
  rw [← isNilpotent_range_toEnd_iff]
  /-
    R : Type u₁
    L : Type u₂
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : IsNoetherian R L
    M : Type u_1
    _i1 : AddCommGroup M
    _i2 : Module R M
    _i3 : LieRingModule L M
    _i4 : LieModule R L M
    h : ∀ (x : L), _root_.IsNilpotent ((LieModule.toEnd R L M) x)
    ⊢ LieModule.IsNilpotent R (Subtype fun x => Membership.mem (LieModule.toEnd R  …
  -/
  let L' := (toEnd R L M).range
  replace h : ∀ y : L', _root_.IsNilpotent (y : Module.End R M) := by
    rintro ⟨-, ⟨y, rfl⟩⟩
    simp [h]
  /-
    R : Type u₁
    L : Type u₂
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : IsNoetherian R L
    M : Type u_1
    _i1 : AddCommGroup M
    _i2 : Module R M
    _i3 : LieRingModule L M
    _i4 : LieModule R L M
    L' : LieSubalgebra R (Module.End R M) := (LieModule.toEnd R L M).range
    h : ∀ (y : Subtype fun x => Membership.mem L' x), _root_.IsNilpotent ↑y
    ⊢ LieModule.IsNilpotent R (Subtype fun x => Membership.mem (LieModule.toEnd R  …
  -/
  change LieModule.IsNilpotent R L' M
  /-
    R : Type u₁
    L : Type u₂
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : IsNoetherian R L
    M : Type u_1
    _i1 : AddCommGroup M
    _i2 : Module R M
    _i3 : LieRingModule L M
    _i4 : LieModule R L M
    L' : LieSubalgebra R (Module.End R M) := (LieModule.toEnd R L M).range
    h : ∀ (y : Subtype fun x => Membership.mem L' x), _root_.IsNilpotent ↑y
    ⊢ LieModule.IsNilpotent R (Subtype fun x => Membership.mem L' x) M
  -/
  let s := {K : LieSubalgebra R L' | LieAlgebra.IsEngelian R K}
  /-
    R : Type u₁
    L : Type u₂
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : IsNoetherian R L
    M : Type u_1
    _i1 : AddCommGroup M
    _i2 : Module R M
    _i3 : LieRingModule L M
    _i4 : LieModule R L M
    L' : LieSubalgebra R (Module.End R M) := (LieModule.toEnd R L M).range
    h : ∀ (y : Subtype fun x => Membership.mem L' x), _root_.IsNilpotent ↑y
    s : Set (LieSubalgebra R (Subtype fun x => Membership.mem L' x)) := setOf fun  …
    ⊢ LieModule.IsNilpotent R (Subtype fun x => Membership.mem L' x) M
  -/
  have hs : s.Nonempty := ⟨⊥, LieAlgebra.isEngelian_of_subsingleton⟩
  suffices ⊤ ∈ s by
    rw [← isNilpotent_of_top_iff]
    apply this M
    simp [LieSubalgebra.toEnd_eq, h]
  have : ∀ K ∈ s, K ≠ ⊤ → ∃ K' ∈ s, K < K' := by
    rintro K (hK₁ : LieAlgebra.IsEngelian R K) hK₂
    apply LieAlgebra.exists_engelian_lieSubalgebra_of_lt_normalizer hK₁
    apply lt_of_le_of_ne K.le_normalizer
    rw [Ne, eq_comm, K.normalizer_eq_self_iff, ← Ne, ←
      LieSubmodule.nontrivial_iff_ne_bot R K]
    have : Nontrivial (L' ⧸ K.toLieSubmodule) := by
      replace hK₂ : K.toLieSubmodule ≠ ⊤ := by
        rwa [Ne, ← LieSubmodule.toSubmodule_inj, K.coe_toLieSubmodule,
          LieSubmodule.top_toSubmodule, ← LieSubalgebra.top_toSubmodule,
          K.toSubmodule_inj]
      exact Submodule.Quotient.nontrivial_of_lt_top _ hK₂.lt_top
    have : LieModule.IsNilpotent R K (L' ⧸ K.toLieSubmodule) := by
      -- Porting note: was refine' hK₁ _ fun x => _
      apply hK₁
      intro x
      have hx := LieAlgebra.isNilpotent_ad_of_isNilpotent (h x)
      apply Module.End.IsNilpotent.mapQ ?_ hx
      -- Porting note: mathlib3 solved this on its own with `submodule.mapq_linear._proof_5`
      intro X HX
      simp only [LieSubalgebra.coe_toLieSubmodule, LieSubalgebra.mem_toSubmodule] at HX
      simp only [LieSubalgebra.coe_toLieSubmodule, Submodule.mem_comap, ad_apply,
        LieSubalgebra.mem_toSubmodule]
      exact LieSubalgebra.lie_mem K x.prop HX
    exact nontrivial_max_triv_of_isNilpotent R K (L' ⧸ K.toLieSubmodule)
  haveI _i5 : IsNoetherian R L' := by
    -- Porting note: was
    -- isNoetherian_of_surjective L _ (LinearMap.range_rangeRestrict (toEnd R L M))
    -- abusing the relation between `LieHom.rangeRestrict` and `LinearMap.rangeRestrict`
    refine isNoetherian_of_surjective L (LieHom.rangeRestrict (toEnd R L M)) ?_
    simp only [LieHom.range_toSubmodule, LieHom.coe_toLinearMap,
      LinearMap.range_eq_top]
    exact LieHom.surjective_rangeRestrict (toEnd R L M)
  /-
    R : Type u₁
    L : Type u₂
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : IsNoetherian R L
    M : Type u_1
    _i1 : AddCommGroup M
    _i2 : Module R M
    _i3 : LieRingModule L M
    _i4 : LieModule R L M
    L' : LieSubalgebra R (Module.End R M) := (LieModule.toEnd R L M).range
    h : ∀ (y : Subtype fun x => Membership.mem L' x), _root_.IsNilpotent ↑y
    s : Set (LieSubalgebra R (Subtype fun x => Membership.mem L' x)) := setOf fun  …
    hs : s.Nonempty
    this : ∀ (K : LieSubalgebra R (Subtype fun x => Membership.mem L' x)), Members …
    _i5 : IsNoetherian R (Subtype fun x => Membership.mem L' x)
    ⊢ Membership.mem s Top.top
  -/
  obtain ⟨K, hK₁, hK₂⟩ := (LieSubalgebra.wellFoundedGT_of_noetherian R L').wf.has_min s hs
  have hK₃ : K = ⊤ := by
    by_contra contra
    obtain ⟨K', hK'₁, hK'₂⟩ := this K hK₁ contra
    exact hK₂ K' hK'₁ hK'₂
  /-
    case intro.intro
    R : Type u₁
    L : Type u₂
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : IsNoetherian R L
    M : Type u_1
    _i1 : AddCommGroup M
    _i2 : Module R M
    _i3 : LieRingModule L M
    _i4 : LieModule R L M
    L' : LieSubalgebra R (Module.End R M) := (LieModule.toEnd R L M).range
    h : ∀ (y : Subtype fun x => Membership.mem L' x), _root_.IsNilpotent ↑y
    s : Set (LieSubalgebra R (Subtype fun x => Membership.mem L' x)) := setOf fun  …
    hs : s.Nonempty
    this : ∀ (K : LieSubalgebra R (Subtype fun x => Membership.mem L' x)), Members …
    _i5 : IsNoetherian R (Subtype fun x => Membership.mem L' x)
    K : LieSubalgebra R (Subtype fun x => Membership.mem L' x)
    hK₁ : Membership.mem s K
    hK₂ : ∀ (x : LieSubalgebra R (Subtype fun x => Membership.mem L' x)), Membersh …
    hK₃ : Eq K Top.top
    ⊢ Membership.mem s Top.top
  -/
  exact hK₃ ▸ hK₁
  /-
    🎉 no goals
  -/


/-- Engel's theorem.

See also `LieModule.isNilpotent_iff_forall'` which assumes that `M` is Noetherian instead of `L`. -/
theorem LieModule.isNilpotent_iff_forall [IsNoetherian R L] :
    LieModule.IsNilpotent R L M ↔ ∀ x, _root_.IsNilpotent <| toEnd R L M x :=
  ⟨fun _ ↦ isNilpotent_toEnd_of_isNilpotent R L M,
   fun h => LieAlgebra.isEngelian_of_isNoetherian M h⟩


/-- Engel's theorem. -/
theorem LieModule.isNilpotent_iff_forall' [IsNoetherian R M] :
    LieModule.IsNilpotent R L M ↔ ∀ x, _root_.IsNilpotent <| toEnd R L M x := by
  /-
    R : Type u₁
    L : Type u₂
    M : Type u₄
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : IsNoetherian R M
    ⊢ Iff (LieModule.IsNilpotent R L M) (∀ (x : L), _root_.IsNilpotent ((LieModule …
  -/
  rw [← isNilpotent_range_toEnd_iff, LieModule.isNilpotent_iff_forall]; simp
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


/-- Engel's theorem. -/
theorem LieAlgebra.isNilpotent_iff_forall [IsNoetherian R L] :
    LieAlgebra.IsNilpotent R L ↔ ∀ x, _root_.IsNilpotent <| LieAlgebra.ad R L x :=
  LieModule.isNilpotent_iff_forall


