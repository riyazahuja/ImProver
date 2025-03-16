/-- Map a pair of submodules under a bilinear map.

This is the submodule version of `Set.image2`. -/
def map₂ (f : M →ₗ[R] N →ₗ[R] P) (p : Submodule R M) (q : Submodule R N) : Submodule R P :=
  ⨆ s : p, q.map (f s)


theorem apply_mem_map₂ (f : M →ₗ[R] N →ₗ[R] P) {m : M} {n : N} {p : Submodule R M}
    {q : Submodule R N} (hm : m ∈ p) (hn : n ∈ q) : f m n ∈ map₂ f p q :=
                                                  /-
                                                    R : Type u_1
                                                    M : Type u_2
                                                    N : Type u_3
                                                    P : Type u_4
                                                    inst✝⁶ : CommSemiring R
                                                    inst✝⁵ : AddCommMonoid M
                                                    inst✝⁴ : AddCommMonoid N
                                                    inst✝³ : AddCommMonoid P
                                                    inst✝² : Module R M
                                                    inst✝¹ : Module R N
                                                    inst✝ : Module R P
                                                    f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N P)
                                                    m : M
                                                    n : N
                                                    p : Submodule R M
                                                    q : Submodule R N
                                                    hm : Membership.mem p m
                                                    hn : Membership.mem q n
                                                    ⊢ Eq ((f ↑⟨m, hm⟩) n) ((f m) n)
                                                  -/
  (le_iSup _ ⟨m, hm⟩ : _ ≤ map₂ f p q) ⟨n, hn, by rfl⟩
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem map₂_le {f : M →ₗ[R] N →ₗ[R] P} {p : Submodule R M} {q : Submodule R N}
    {r : Submodule R P} : map₂ f p q ≤ r ↔ ∀ m ∈ p, ∀ n ∈ q, f m n ∈ r :=
  ⟨fun H _m hm _n hn => H <| apply_mem_map₂ _ hm hn, fun H =>
    iSup_le fun ⟨m, hm⟩ => map_le_iff_le_comap.2 fun n hn => H m hm n hn⟩


theorem map₂_span_span (f : M →ₗ[R] N →ₗ[R] P) (s : Set M) (t : Set N) :
    map₂ f (span R s) (span R t) = span R (Set.image2 (fun m n => f m n) s t) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁶ : CommSemiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid N
    inst✝³ : AddCommMonoid P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N P)
    s : Set M
    t : Set N
    ⊢ Eq (Submodule.map₂ f (Submodule.span R s) (Submodule.span R t)) (Submodule.s …
  -/
  apply le_antisymm
    /-
      case a
      R : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid N
      inst✝³ : AddCommMonoid P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N P)
      s : Set M
      t : Set N
      ⊢ LE.le (Submodule.map₂ f (Submodule.span R s) (Submodule.span R t)) (Submodul …
    -/
  · rw [map₂_le]
    /-
      case a
      R : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid N
      inst✝³ : AddCommMonoid P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N P)
      s : Set M
      t : Set N
      ⊢ ∀ (m : M), Membership.mem (Submodule.span R s) m → ∀ (n : N), Membership.mem …
    -/
    apply @span_induction R M _ _ _ s
    on_goal 1 =>
      intro a ha
      apply @span_induction R N _ _ _ t
      · intro b hb
        exact subset_span ⟨_, ‹_›, _, ‹_›, rfl⟩
    all_goals
      intros
      simp only [*, add_mem, smul_mem, zero_mem, _root_.map_zero, map_add,
        LinearMap.zero_apply, LinearMap.add_apply, LinearMap.smul_apply, map_smul]
    /-
      case a
      R : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid N
      inst✝³ : AddCommMonoid P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N P)
      s : Set M
      t : Set N
      ⊢ LE.le (Submodule.span R (Set.image2 (fun m n => (f m) n) s t)) (Submodule.ma …
    -/
  · rw [span_le, image2_subset_iff]
    /-
      case a
      R : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid N
      inst✝³ : AddCommMonoid P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N P)
      s : Set M
      t : Set N
      ⊢ ∀ (x : M), Membership.mem s x → ∀ (y : N), Membership.mem t y → Membership.m …
    -/
    intro a ha b hb
    /-
      case a
      R : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid N
      inst✝³ : AddCommMonoid P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N P)
      s : Set M
      t : Set N
      a : M
      ha : Membership.mem s a
      b : N
      hb : Membership.mem t b
      ⊢ Membership.mem (↑(Submodule.map₂ f (Submodule.span R s) (Submodule.span R t) …
    -/
    exact apply_mem_map₂ _ (subset_span ha) (subset_span hb)
    /-
      🎉 no goals
    -/

@[simp]
theorem map₂_bot_right (f : M →ₗ[R] N →ₗ[R] P) (p : Submodule R M) : map₂ f p ⊥ = ⊥ :=
  eq_bot_iff.2 <|
    map₂_le.2 fun m _hm n hn => by
      /-
        R : Type u_1
        M : Type u_2
        N : Type u_3
        P : Type u_4
        inst✝⁶ : CommSemiring R
        inst✝⁵ : AddCommMonoid M
        inst✝⁴ : AddCommMonoid N
        inst✝³ : AddCommMonoid P
        inst✝² : Module R M
        inst✝¹ : Module R N
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N P)
        p : Submodule R M
        m : M
        _hm : Membership.mem p m
        n : N
        hn : Membership.mem Bot.bot n
        ⊢ Membership.mem Bot.bot ((f m) n)
      -/
      rw [Submodule.mem_bot] at hn
      /-
        R : Type u_1
        M : Type u_2
        N : Type u_3
        P : Type u_4
        inst✝⁶ : CommSemiring R
        inst✝⁵ : AddCommMonoid M
        inst✝⁴ : AddCommMonoid N
        inst✝³ : AddCommMonoid P
        inst✝² : Module R M
        inst✝¹ : Module R N
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N P)
        p : Submodule R M
        m : M
        _hm : Membership.mem p m
        n : N
        hn : Eq n 0
        ⊢ Membership.mem Bot.bot ((f m) n)
      -/
      rw [hn, LinearMap.map_zero]; simp only [mem_bot]
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
theorem map₂_bot_left (f : M →ₗ[R] N →ₗ[R] P) (q : Submodule R N) : map₂ f ⊥ q = ⊥ :=
  eq_bot_iff.2 <|
    map₂_le.2 fun m hm n _ => by
      /-
        R : Type u_1
        M : Type u_2
        N : Type u_3
        P : Type u_4
        inst✝⁶ : CommSemiring R
        inst✝⁵ : AddCommMonoid M
        inst✝⁴ : AddCommMonoid N
        inst✝³ : AddCommMonoid P
        inst✝² : Module R M
        inst✝¹ : Module R N
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N P)
        q : Submodule R N
        m : M
        hm : Membership.mem Bot.bot m
        n : N
        x✝ : Membership.mem q n
        ⊢ Membership.mem Bot.bot ((f m) n)
      -/
      rw [Submodule.mem_bot] at hm ⊢
      /-
        R : Type u_1
        M : Type u_2
        N : Type u_3
        P : Type u_4
        inst✝⁶ : CommSemiring R
        inst✝⁵ : AddCommMonoid M
        inst✝⁴ : AddCommMonoid N
        inst✝³ : AddCommMonoid P
        inst✝² : Module R M
        inst✝¹ : Module R N
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N P)
        q : Submodule R N
        m : M
        hm : Eq m 0
        n : N
        x✝ : Membership.mem q n
        ⊢ Eq ((f m) n) 0
      -/
      rw [hm, LinearMap.map_zero₂]
      /-
        🎉 no goals
      -/


@[gcongr, mono]
theorem map₂_le_map₂ {f : M →ₗ[R] N →ₗ[R] P} {p₁ p₂ : Submodule R M} {q₁ q₂ : Submodule R N}
    (hp : p₁ ≤ p₂) (hq : q₁ ≤ q₂) : map₂ f p₁ q₁ ≤ map₂ f p₂ q₂ :=
  map₂_le.2 fun _m hm _n hn => apply_mem_map₂ _ (hp hm) (hq hn)


theorem map₂_le_map₂_left {f : M →ₗ[R] N →ₗ[R] P} {p₁ p₂ : Submodule R M} {q : Submodule R N}
    (h : p₁ ≤ p₂) : map₂ f p₁ q ≤ map₂ f p₂ q :=
  map₂_le_map₂ h (le_refl q)


theorem map₂_le_map₂_right {f : M →ₗ[R] N →ₗ[R] P} {p : Submodule R M} {q₁ q₂ : Submodule R N}
    (h : q₁ ≤ q₂) : map₂ f p q₁ ≤ map₂ f p q₂ :=
  map₂_le_map₂ (le_refl p) h


theorem map₂_sup_right (f : M →ₗ[R] N →ₗ[R] P) (p : Submodule R M) (q₁ q₂ : Submodule R N) :
    map₂ f p (q₁ ⊔ q₂) = map₂ f p q₁ ⊔ map₂ f p q₂ :=
  le_antisymm
    (map₂_le.2 fun _m hm _np hnp =>
      let ⟨_n, hn, _p, hp, hnp⟩ := mem_sup.1 hnp
      mem_sup.2 ⟨_, apply_mem_map₂ _ hm hn, _, apply_mem_map₂ _ hm hp, hnp ▸ (map_add _ _ _).symm⟩)
    (sup_le (map₂_le_map₂_right le_sup_left) (map₂_le_map₂_right le_sup_right))


theorem map₂_sup_left (f : M →ₗ[R] N →ₗ[R] P) (p₁ p₂ : Submodule R M) (q : Submodule R N) :
    map₂ f (p₁ ⊔ p₂) q = map₂ f p₁ q ⊔ map₂ f p₂ q :=
  le_antisymm
    (map₂_le.2 fun _mn hmn _p hp =>
      let ⟨_m, hm, _n, hn, hmn⟩ := mem_sup.1 hmn
      mem_sup.2
        ⟨_, apply_mem_map₂ _ hm hp, _, apply_mem_map₂ _ hn hp,
          hmn ▸ (LinearMap.map_add₂ _ _ _ _).symm⟩)
    (sup_le (map₂_le_map₂_left le_sup_left) (map₂_le_map₂_left le_sup_right))


theorem image2_subset_map₂ (f : M →ₗ[R] N →ₗ[R] P) (p : Submodule R M) (q : Submodule R N) :
    Set.image2 (fun m n => f m n) (↑p : Set M) (↑q : Set N) ⊆ (↑(map₂ f p q) : Set P) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁶ : CommSemiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid N
    inst✝³ : AddCommMonoid P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N P)
    p : Submodule R M
    q : Submodule R N
    ⊢ HasSubset.Subset (Set.image2 (fun m n => (f m) n) ↑p ↑q) ↑(Submodule.map₂ f  …
  -/
  rintro _ ⟨i, hi, j, hj, rfl⟩
  /-
    case intro.intro.intro.intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁶ : CommSemiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid N
    inst✝³ : AddCommMonoid P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N P)
    p : Submodule R M
    q : Submodule R N
    i : M
    hi : Membership.mem (↑p) i
    j : N
    hj : Membership.mem (↑q) j
    ⊢ Membership.mem (↑(Submodule.map₂ f p q)) ((fun m n => (f m) n) i j)
  -/
  exact apply_mem_map₂ _ hi hj
  /-
    🎉 no goals
  -/


theorem map₂_eq_span_image2 (f : M →ₗ[R] N →ₗ[R] P) (p : Submodule R M) (q : Submodule R N) :
    map₂ f p q = span R (Set.image2 (fun m n => f m n) (p : Set M) (q : Set N)) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁶ : CommSemiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid N
    inst✝³ : AddCommMonoid P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N P)
    p : Submodule R M
    q : Submodule R N
    ⊢ Eq (Submodule.map₂ f p q) (Submodule.span R (Set.image2 (fun m n => (f m) n) …
  -/
  rw [← map₂_span_span, span_eq, span_eq]
  /-
    🎉 no goals
  -/


theorem map₂_flip (f : M →ₗ[R] N →ₗ[R] P) (p : Submodule R M) (q : Submodule R N) :
    map₂ f.flip q p = map₂ f p q := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁶ : CommSemiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid N
    inst✝³ : AddCommMonoid P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N P)
    p : Submodule R M
    q : Submodule R N
    ⊢ Eq (Submodule.map₂ f.flip q p) (Submodule.map₂ f p q)
  -/
  rw [map₂_eq_span_image2, map₂_eq_span_image2, Set.image2_swap]
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁶ : CommSemiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid N
    inst✝³ : AddCommMonoid P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N P)
    p : Submodule R M
    q : Submodule R N
    ⊢ Eq (Submodule.span R (Set.image2 (fun a b => (f.flip b) a) ↑p ↑q)) (Submodul …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem map₂_iSup_left (f : M →ₗ[R] N →ₗ[R] P) (s : ι → Submodule R M) (t : Submodule R N) :
    map₂ f (⨆ i, s i) t = ⨆ i, map₂ f (s i) t := by
  suffices map₂ f (⨆ i, span R (s i : Set M)) (span R t) = ⨆ i, map₂ f (span R (s i)) (span R t) by
    simpa only [span_eq] using this
  /-
    ι : Sort uι
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁶ : CommSemiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid N
    inst✝³ : AddCommMonoid P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N P)
    s : ι → Submodule R M
    t : Submodule R N
    ⊢ Eq (Submodule.map₂ f (iSup fun i => Submodule.span R ↑(s i)) (Submodule.span …
  -/
  simp_rw [map₂_span_span, ← span_iUnion, map₂_span_span, Set.image2_iUnion_left]
  /-
    🎉 no goals
  -/


theorem map₂_iSup_right (f : M →ₗ[R] N →ₗ[R] P) (s : Submodule R M) (t : ι → Submodule R N) :
    map₂ f s (⨆ i, t i) = ⨆ i, map₂ f s (t i) := by
  suffices map₂ f (span R s) (⨆ i, span R (t i : Set N)) = ⨆ i, map₂ f (span R s) (span R (t i)) by
    simpa only [span_eq] using this
  /-
    ι : Sort uι
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁶ : CommSemiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid N
    inst✝³ : AddCommMonoid P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N P)
    s : Submodule R M
    t : ι → Submodule R N
    ⊢ Eq (Submodule.map₂ f (Submodule.span R ↑s) (iSup fun i => Submodule.span R ↑ …
  -/
  simp_rw [map₂_span_span, ← span_iUnion, map₂_span_span, Set.image2_iUnion_right]
  /-
    🎉 no goals
  -/


theorem map₂_span_singleton_eq_map (f : M →ₗ[R] N →ₗ[R] P) (m : M) :
    map₂ f (span R {m}) = map (f m) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁶ : CommSemiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid N
    inst✝³ : AddCommMonoid P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N P)
    m : M
    ⊢ Eq (Submodule.map₂ f (Submodule.span R (Singleton.singleton m))) (Submodule. …
  -/
  funext s
  /-
    case h
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁶ : CommSemiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid N
    inst✝³ : AddCommMonoid P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N P)
    m : M
    s : Submodule R N
    ⊢ Eq (Submodule.map₂ f (Submodule.span R (Singleton.singleton m)) s) (Submodul …
  -/
  rw [← span_eq s, map₂_span_span, image2_singleton_left, map_span]
  /-
    🎉 no goals
  -/


theorem map₂_span_singleton_eq_map_flip (f : M →ₗ[R] N →ₗ[R] P) (s : Submodule R M) (n : N) :
                                                   /-
                                                     R : Type u_1
                                                     M : Type u_2
                                                     N : Type u_3
                                                     P : Type u_4
                                                     inst✝⁶ : CommSemiring R
                                                     inst✝⁵ : AddCommMonoid M
                                                     inst✝⁴ : AddCommMonoid N
                                                     inst✝³ : AddCommMonoid P
                                                     inst✝² : Module R M
                                                     inst✝¹ : Module R N
                                                     inst✝ : Module R P
                                                     f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N P)
                                                     s : Submodule R M
                                                     n : N
                                                     ⊢ Eq (Submodule.map₂ f s (Submodule.span R (Singleton.singleton n))) (Submodul …
                                                   -/
    map₂ f s (span R {n}) = map (f.flip n) s := by rw [← map₂_span_singleton_eq_map, map₂_flip]
                                                   /-
                                                     🎉 no goals
                                                   -/


