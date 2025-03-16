/-- `IsNoetherian R M` is the proposition that `M` is a Noetherian `R`-module,
implemented as the predicate that all `R`-submodules of `M` are finitely generated.
-/
-- Porting note: should this be renamed to `Noetherian`?
class IsNoetherian (R M) [Semiring R] [AddCommMonoid M] [Module R M] : Prop where
  noetherian : ∀ s : Submodule R M, s.FG


/-- An R-module is Noetherian iff all its submodules are finitely-generated. -/
theorem isNoetherian_def : IsNoetherian R M ↔ ∀ s : Submodule R M, s.FG :=
  ⟨fun h => h.noetherian, IsNoetherian.mk⟩


theorem isNoetherian_submodule {N : Submodule R M} :
    IsNoetherian R N ↔ ∀ s : Submodule R M, s ≤ N → s.FG := by
  refine ⟨fun ⟨hn⟩ => fun s hs =>
    have : s ≤ LinearMap.range N.subtype := N.range_subtype.symm ▸ hs
    Submodule.map_comap_eq_self this ▸ (hn _).map _,
    fun h => ⟨fun s => ?_⟩⟩
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    N : Submodule R M
    h : ∀ (s : Submodule R M), LE.le s N → s.FG
    s : Submodule R (Subtype fun x => Membership.mem N x)
    ⊢ s.FG
  -/
  have f := (Submodule.equivMapOfInjective N.subtype Subtype.val_injective s).symm
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    N : Submodule R M
    h : ∀ (s : Submodule R M), LE.le s N → s.FG
    s : Submodule R (Subtype fun x => Membership.mem N x)
    f : LinearEquiv (RingHom.id R) (Subtype fun x => Membership.mem (Submodule.map …
    ⊢ s.FG
  -/
  have h₁ := h (s.map N.subtype) (Submodule.map_subtype_le N s)
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    N : Submodule R M
    h : ∀ (s : Submodule R M), LE.le s N → s.FG
    s : Submodule R (Subtype fun x => Membership.mem N x)
    f : LinearEquiv (RingHom.id R) (Subtype fun x => Membership.mem (Submodule.map …
    h₁ : (Submodule.map N.subtype s).FG
    ⊢ s.FG
  -/
  have h₂ : (⊤ : Submodule R (s.map N.subtype)).map f = ⊤ := by simp
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    N : Submodule R M
    h : ∀ (s : Submodule R M), LE.le s N → s.FG
    s : Submodule R (Subtype fun x => Membership.mem N x)
    f : LinearEquiv (RingHom.id R) (Subtype fun x => Membership.mem (Submodule.map …
    h₁ : (Submodule.map N.subtype s).FG
    h₂ : Eq (Submodule.map f Top.top) Top.top
    ⊢ s.FG
  -/
  have h₃ := ((Submodule.fg_top _).2 h₁).map (↑f : _ →ₗ[R] s)
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    N : Submodule R M
    h : ∀ (s : Submodule R M), LE.le s N → s.FG
    s : Submodule R (Subtype fun x => Membership.mem N x)
    f : LinearEquiv (RingHom.id R) (Subtype fun x => Membership.mem (Submodule.map …
    h₁ : (Submodule.map N.subtype s).FG
    h₂ : Eq (Submodule.map f Top.top) Top.top
    h₃ : (Submodule.map (↑f) Top.top).FG
    ⊢ s.FG
  -/
  exact (Submodule.fg_top _).1 (h₂ ▸ h₃)
  /-
    🎉 no goals
  -/


theorem isNoetherian_submodule_left {N : Submodule R M} :
    IsNoetherian R N ↔ ∀ s : Submodule R M, (N ⊓ s).FG :=
  isNoetherian_submodule.trans ⟨fun H _ => H _ inf_le_left, fun H _ hs => inf_of_le_right hs ▸ H _⟩


theorem isNoetherian_submodule_right {N : Submodule R M} :
    IsNoetherian R N ↔ ∀ s : Submodule R M, (s ⊓ N).FG :=
  isNoetherian_submodule.trans ⟨fun H _ => H _ inf_le_right, fun H _ hs => inf_of_le_left hs ▸ H _⟩


instance isNoetherian_submodule' [IsNoetherian R M] (N : Submodule R M) : IsNoetherian R N :=
  isNoetherian_submodule.2 fun _ _ => IsNoetherian.noetherian _


theorem isNoetherian_of_le {s t : Submodule R M} [ht : IsNoetherian R t] (h : s ≤ t) :
    IsNoetherian R s :=
  isNoetherian_submodule.mpr fun _ hs' => isNoetherian_submodule.mp ht _ (le_trans hs' h)


theorem isNoetherian_iff' : IsNoetherian R M ↔ WellFoundedGT (Submodule R M) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ⊢ Iff (IsNoetherian R M) (WellFoundedGT (Submodule R M))
  -/
  have := (CompleteLattice.wellFoundedGT_characterisations <| Submodule R M).out 0 3
  -- Porting note: inlining this makes rw complain about it being a metavariable
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    this : Iff (WellFoundedGT (Submodule R M)) (∀ (k : Submodule R M), CompleteLat …
    ⊢ Iff (IsNoetherian R M) (WellFoundedGT (Submodule R M))
  -/
  rw [this]
  exact
    ⟨fun ⟨h⟩ => fun k => (fg_iff_compact k).mp (h k), fun h =>
      ⟨fun k => (fg_iff_compact k).mpr (h k)⟩⟩


theorem isNoetherian_iff :
    IsNoetherian R M ↔ WellFounded ((· > ·) : Submodule R M → Submodule R M → Prop) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ⊢ Iff (IsNoetherian R M) (WellFounded fun x1 x2 => GT.gt x1 x2)
  -/
  rw [isNoetherian_iff', ← isWellFounded_iff]
  /-
    🎉 no goals
  -/


alias ⟨IsNoetherian.wf, _⟩ := isNoetherian_iff


alias ⟨IsNoetherian.wellFoundedGT, isNoetherian_mk⟩ := isNoetherian_iff'


instance wellFoundedGT [h : IsNoetherian R M] : WellFoundedGT (Submodule R M) :=
  h.wellFoundedGT


theorem isNoetherian_iff_fg_wellFounded :
    IsNoetherian R M ↔ WellFoundedGT { N : Submodule R M // N.FG } := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ⊢ Iff (IsNoetherian R M) (WellFoundedGT (Subtype fun N => N.FG))
  -/
  let α := { N : Submodule R M // N.FG }
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    α : Type (max 0 u_2) := Subtype fun N => N.FG
    ⊢ Iff (IsNoetherian R M) (WellFoundedGT (Subtype fun N => N.FG))
  -/
  constructor
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      α : Type (max 0 u_2) := Subtype fun N => N.FG
      ⊢ IsNoetherian R M → WellFoundedGT (Subtype fun N => N.FG)
    -/
  · intro H
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      α : Type (max 0 u_2) := Subtype fun N => N.FG
      H : IsNoetherian R M
      ⊢ WellFoundedGT (Subtype fun N => N.FG)
    -/
    let f : α ↪o Submodule R M := OrderEmbedding.subtype _
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      α : Type (max 0 u_2) := Subtype fun N => N.FG
      H : IsNoetherian R M
      f : OrderEmbedding α (Submodule R M) := OrderEmbedding.subtype fun N => N.FG
      ⊢ WellFoundedGT (Subtype fun N => N.FG)
    -/
    exact OrderEmbedding.wellFoundedLT f.dual
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      α : Type (max 0 u_2) := Subtype fun N => N.FG
      ⊢ WellFoundedGT (Subtype fun N => N.FG) → IsNoetherian R M
    -/
  · intro H
    /-
      case mpr
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      α : Type (max 0 u_2) := Subtype fun N => N.FG
      H : WellFoundedGT (Subtype fun N => N.FG)
      ⊢ IsNoetherian R M
    -/
    constructor
    /-
      case mpr.noetherian
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      α : Type (max 0 u_2) := Subtype fun N => N.FG
      H : WellFoundedGT (Subtype fun N => N.FG)
      ⊢ ∀ (s : Submodule R M), s.FG
    -/
    intro N
    obtain ⟨⟨N₀, h₁⟩, e : N₀ ≤ N, h₂⟩ :=
      WellFounded.has_min H.wf { N' : α | N'.1 ≤ N } ⟨⟨⊥, Submodule.fg_bot⟩, @bot_le _ _ _ N⟩
    /-
      case mpr.noetherian.intro.mk.intro
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      α : Type (max 0 u_2) := Subtype fun N => N.FG
      H : WellFoundedGT (Subtype fun N => N.FG)
      N N₀ : Submodule R M
      h₁ : N₀.FG
      e : LE.le N₀ N
      h₂ : ∀ (x : Subtype fun N => N.FG), Membership.mem (setOf fun N' => LE.le (↑N' …
      ⊢ N.FG
    -/
    convert h₁
    /-
      case h.e'_6
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      α : Type (max 0 u_2) := Subtype fun N => N.FG
      H : WellFoundedGT (Subtype fun N => N.FG)
      N N₀ : Submodule R M
      h₁ : N₀.FG
      e : LE.le N₀ N
      h₂ : ∀ (x : Subtype fun N => N.FG), Membership.mem (setOf fun N' => LE.le (↑N' …
      ⊢ Eq N N₀
    -/
    refine (e.antisymm ?_).symm
    /-
      case h.e'_6
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      α : Type (max 0 u_2) := Subtype fun N => N.FG
      H : WellFoundedGT (Subtype fun N => N.FG)
      N N₀ : Submodule R M
      h₁ : N₀.FG
      e : LE.le N₀ N
      h₂ : ∀ (x : Subtype fun N => N.FG), Membership.mem (setOf fun N' => LE.le (↑N' …
      ⊢ LE.le N N₀
    -/
    by_contra h₃
    /-
      case h.e'_6
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      α : Type (max 0 u_2) := Subtype fun N => N.FG
      H : WellFoundedGT (Subtype fun N => N.FG)
      N N₀ : Submodule R M
      h₁ : N₀.FG
      e : LE.le N₀ N
      h₂ : ∀ (x : Subtype fun N => N.FG), Membership.mem (setOf fun N' => LE.le (↑N' …
      h₃ : Not (LE.le N N₀)
      ⊢ False
    -/
    obtain ⟨x, hx₁ : x ∈ N, hx₂ : x ∉ N₀⟩ := Set.not_subset.mp h₃
    /-
      case h.e'_6.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      α : Type (max 0 u_2) := Subtype fun N => N.FG
      H : WellFoundedGT (Subtype fun N => N.FG)
      N N₀ : Submodule R M
      h₁ : N₀.FG
      e : LE.le N₀ N
      h₂ : ∀ (x : Subtype fun N => N.FG), Membership.mem (setOf fun N' => LE.le (↑N' …
      h₃ : Not (LE.le N N₀)
      x : M
      hx₁ : Membership.mem N x
      hx₂ : Not (Membership.mem N₀ x)
      ⊢ False
    -/
    apply hx₂
    rw [eq_of_le_of_not_lt (le_sup_right : N₀ ≤ _) (h₂
      ⟨_, Submodule.FG.sup ⟨{x}, by rw [Finset.coe_singleton]⟩ h₁⟩ <|
      sup_le ((Submodule.span_singleton_le_iff_mem _ _).mpr hx₁) e)]
    /-
      case h.e'_6.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      α : Type (max 0 u_2) := Subtype fun N => N.FG
      H : WellFoundedGT (Subtype fun N => N.FG)
      N N₀ : Submodule R M
      h₁ : N₀.FG
      e : LE.le N₀ N
      h₂ : ∀ (x : Subtype fun N => N.FG), Membership.mem (setOf fun N' => LE.le (↑N' …
      h₃ : Not (LE.le N N₀)
      x : M
      hx₁ : Membership.mem N x
      hx₂ : Not (Membership.mem N₀ x)
      ⊢ Membership.mem (Max.max (Submodule.span R (Singleton.singleton x)) N₀) x
    -/
    exact (le_sup_left : (R ∙ x) ≤ _) (Submodule.mem_span_singleton_self _)
    /-
      🎉 no goals
    -/


/-- A module is Noetherian iff every nonempty set of submodules has a maximal submodule among them.
-/
theorem set_has_maximal_iff_noetherian :
    (∀ a : Set <| Submodule R M, a.Nonempty → ∃ M' ∈ a, ∀ I ∈ a, ¬M' < I) ↔ IsNoetherian R M := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ⊢ Iff (∀ (a : Set (Submodule R M)), a.Nonempty → Exists fun M' => And (Members …
  -/
  rw [isNoetherian_iff, WellFounded.wellFounded_iff_has_min]
  /-
    🎉 no goals
  -/


/-- A module is Noetherian iff every increasing chain of submodules stabilizes. -/
theorem monotone_stabilizes_iff_noetherian :
    (∀ f : ℕ →o Submodule R M, ∃ n, ∀ m, n ≤ m → f n = f m) ↔ IsNoetherian R M := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ⊢ Iff (∀ (f : OrderHom Nat (Submodule R M)), Exists fun n => ∀ (m : Nat), LE.l …
  -/
  rw [isNoetherian_iff, WellFounded.monotone_chain_condition]
  /-
    🎉 no goals
  -/


/-- For an endomorphism of a Noetherian module, any sufficiently large iterate has disjoint kernel
and range. -/
theorem LinearMap.eventually_disjoint_ker_pow_range_pow (f : M →ₗ[R] M) :
    ∀ᶠ n in atTop, Disjoint (LinearMap.ker (f ^ n)) (LinearMap.range (f ^ n)) := by
  obtain ⟨n, hn : ∀ m, n ≤ m → LinearMap.ker (f ^ n) = LinearMap.ker (f ^ m)⟩ :=
    monotone_stabilizes_iff_noetherian.mpr inferInstance f.iterateKer
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : IsNoetherian R M
    f : LinearMap (RingHom.id R) M M
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (LinearMap.ker (HPow.hPow f n)) (LinearMap.ke …
    ⊢ Filter.Eventually (fun n => Disjoint (LinearMap.ker (HPow.hPow f n)) (Linear …
  -/
  refine eventually_atTop.mpr ⟨n, fun m hm ↦ disjoint_iff.mpr ?_⟩
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : IsNoetherian R M
    f : LinearMap (RingHom.id R) M M
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (LinearMap.ker (HPow.hPow f n)) (LinearMap.ke …
    m : Nat
    hm : GE.ge m n
    ⊢ Eq (Min.min (LinearMap.ker (HPow.hPow f m)) (LinearMap.range (HPow.hPow f m) …
  -/
  rw [← hn _ hm, Submodule.eq_bot_iff]
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : IsNoetherian R M
    f : LinearMap (RingHom.id R) M M
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (LinearMap.ker (HPow.hPow f n)) (LinearMap.ke …
    m : Nat
    hm : GE.ge m n
    ⊢ ∀ (x : M), Membership.mem (Min.min (LinearMap.ker (HPow.hPow f n)) (LinearMa …
  -/
  rintro - ⟨hx, ⟨x, rfl⟩⟩
  /-
    case intro.intro.intro
    R : Type u_1
    M : Type u_2
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : IsNoetherian R M
    f : LinearMap (RingHom.id R) M M
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (LinearMap.ker (HPow.hPow f n)) (LinearMap.ke …
    m : Nat
    hm : GE.ge m n
    x : M
    hx : Membership.mem (↑(LinearMap.ker (HPow.hPow f n))) ((HPow.hPow f m) x)
    ⊢ Eq ((HPow.hPow f m) x) 0
  -/
  apply LinearMap.pow_map_zero_of_le hm
  replace hx : x ∈ LinearMap.ker (f ^ (n + m)) := by
    simpa [f.pow_apply n, f.pow_apply m, ← f.pow_apply (n + m), ← iterate_add_apply] using hx
  /-
    case intro.intro.intro
    R : Type u_1
    M : Type u_2
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : IsNoetherian R M
    f : LinearMap (RingHom.id R) M M
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (LinearMap.ker (HPow.hPow f n)) (LinearMap.ke …
    m : Nat
    hm : GE.ge m n
    x : M
    hx : Membership.mem (LinearMap.ker (HPow.hPow f (HAdd.hAdd n m))) x
    ⊢ Eq ((HPow.hPow f n) x) 0
  -/
  rwa [← hn _ (n.le_add_right m)] at hx
  /-
    🎉 no goals
  -/


lemma LinearMap.eventually_iSup_ker_pow_eq (f : M →ₗ[R] M) :
    ∀ᶠ n in atTop, ⨆ m, LinearMap.ker (f ^ m) = LinearMap.ker (f ^ n) := by
  obtain ⟨n, hn : ∀ m, n ≤ m → ker (f ^ n) = ker (f ^ m)⟩ :=
    monotone_stabilizes_iff_noetherian.mpr inferInstance f.iterateKer
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : IsNoetherian R M
    f : LinearMap (RingHom.id R) M M
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (LinearMap.ker (HPow.hPow f n)) (LinearMap.ke …
    ⊢ Filter.Eventually (fun n => Eq (iSup fun m => LinearMap.ker (HPow.hPow f m)) …
  -/
  refine eventually_atTop.mpr ⟨n, fun m hm ↦ ?_⟩
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : IsNoetherian R M
    f : LinearMap (RingHom.id R) M M
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (LinearMap.ker (HPow.hPow f n)) (LinearMap.ke …
    m : Nat
    hm : GE.ge m n
    ⊢ Eq (iSup fun m => LinearMap.ker (HPow.hPow f m)) (LinearMap.ker (HPow.hPow f …
  -/
  refine le_antisymm (iSup_le fun l ↦ ?_) (le_iSup (fun i ↦ LinearMap.ker (f ^ i)) m)
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : IsNoetherian R M
    f : LinearMap (RingHom.id R) M M
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (LinearMap.ker (HPow.hPow f n)) (LinearMap.ke …
    m : Nat
    hm : GE.ge m n
    l : Nat
    ⊢ LE.le (LinearMap.ker (HPow.hPow f l)) (LinearMap.ker (HPow.hPow f m))
  -/
  rcases le_or_lt m l with h | h
    /-
      case intro.inl
      R : Type u_1
      M : Type u_2
      inst✝³ : Semiring R
      inst✝² : AddCommMonoid M
      inst✝¹ : Module R M
      inst✝ : IsNoetherian R M
      f : LinearMap (RingHom.id R) M M
      n : Nat
      hn : ∀ (m : Nat), LE.le n m → Eq (LinearMap.ker (HPow.hPow f n)) (LinearMap.ke …
      m : Nat
      hm : GE.ge m n
      l : Nat
      h : LE.le m l
      ⊢ LE.le (LinearMap.ker (HPow.hPow f l)) (LinearMap.ker (HPow.hPow f m))
    -/
  · rw [← hn _ (hm.trans h), hn _ hm]
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      R : Type u_1
      M : Type u_2
      inst✝³ : Semiring R
      inst✝² : AddCommMonoid M
      inst✝¹ : Module R M
      inst✝ : IsNoetherian R M
      f : LinearMap (RingHom.id R) M M
      n : Nat
      hn : ∀ (m : Nat), LE.le n m → Eq (LinearMap.ker (HPow.hPow f n)) (LinearMap.ke …
      m : Nat
      hm : GE.ge m n
      l : Nat
      h : LT.lt l m
      ⊢ LE.le (LinearMap.ker (HPow.hPow f l)) (LinearMap.ker (HPow.hPow f m))
    -/
  · exact f.iterateKer.monotone h.le
    /-
      🎉 no goals
    -/


/-- A (semi)ring is Noetherian if it is Noetherian as a module over itself,
i.e. all its ideals are finitely generated. -/
abbrev IsNoetherianRing (R) [Semiring R] :=
  IsNoetherian R R


theorem isNoetherianRing_iff {R} [Semiring R] : IsNoetherianRing R ↔ IsNoetherian R R :=
  Iff.rfl


/-- A ring is Noetherian if and only if all its ideals are finitely-generated. -/
theorem isNoetherianRing_iff_ideal_fg (R : Type*) [Semiring R] :
    IsNoetherianRing R ↔ ∀ I : Ideal R, I.FG :=
  isNoetherianRing_iff.trans isNoetherian_def

